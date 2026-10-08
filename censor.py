#!/usr/bin/env python3
"""
censor.py -- privacy-blur a folder of clips over a user-picked region.

Built for some incidents are in view of a privacy zone
, but not specific to that tool -- it just takes a flat folder of .mp4 clips and a region.json.

Usage:
    # 1. Pick the region once per folder (writes <folder>/region.json)
    python select_region.py --folder <file path>

    # 2. Censor every clip in that folder -> <folder>/censored/<same filename>.mp4
    python censor.py --folder <file path>
    python censor.py --folder "..." --workers 8
    python censor.py --folder "..." --force   # re-render clips already in censored/

Effect: solid black fill over the picked region, hard (crisp) edge -- the
cheapest per-frame op available, a boolean-indexed assignment to 0. Went
through two earlier, more expensive versions (gaussian blur, then a
feathered/soft-edge black fill) before landing here once the actual goal
became "as little per-frame processing as possible" for a 2000+-clip batch.

`--effect blur` brings the gaussian version back as an opt-in (full region,
hard mask edge, same as the black fill's mask -- not the old feathered
version). `--gpu` on top of that runs the blur on CUDA via PyTorch instead
of cv2.GaussianBlur on CPU; construct once per worker process (like
motion_scan.py's TorchMotionDetector), not per clip, since a CUDA context is
worth amortizing across everything one worker handles.

`--effect crop` cuts every frame down to the picked region's bounding box
instead of hiding it -- the region here means the opposite of black/blur's
(the box to KEEP, not hide), so point --region at a separate region.json:
    python select_region.py --folder "..." --region "folder/region_crop.json"
    python censor.py --folder "..." --region "folder/region_crop.json" --effect crop
Fewer pixels to encode beats even black fill's per-pixel op, since libx264
cost scales with pixel count.

Dependencies: opencv-python, numpy, tqdm (same as motion_scan.py), plus
ffmpeg/ffprobe on PATH. torch is optional, only needed for --gpu.
"""
import argparse
import multiprocessing
import os
import queue as queue_mod
import subprocess
import sys
import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import select_region as region_lib

# ---------------------------------------------------------------- config ---
# Solid black fill with a hard (crisp, unfeathered) mask edge -- cheapest
# possible per-frame op: a boolean-indexed assignment to 0, no float
# conversion/blend, no convolution. (Started as a gaussian blur to match a
# reference photo, which was far too slow on real HD footage under many
# concurrent workers -- ~5.5s/frame at --workers 19; then a soft-feathered
# black fill, still allocating a float32 blend per frame; this is the
# no-frills version once the actual goal became "as little per-frame work as
# possible." See DESIGN.md.)
CRF = 22
PRESET = "fast"
PROGRESS_EVERY_N_FRAMES = 5  # throttle per-worker progress updates through the queue
DEFAULT_BLUR_SIGMA = 55.0  # matches the sigma used before the effect was switched to black fill
# -----------------------------------------------------------------------------


def torch_cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


class TorchGaussianBlur:
    """GPU gaussian blur via PyTorch CUDA -- a depthwise separable conv2d
    equivalent to cv2.GaussianBlur(img, (0, 0), sigma). One instance is built
    per worker process and reused across every clip that worker handles (see
    module docstring for why), not rebuilt per clip."""

    def __init__(self, sigma):
        import torch
        self.torch = torch
        self.device = torch.device("cuda")
        radius = max(1, int(round(sigma * 3)))
        self.pad = radius
        xs = torch.arange(radius * 2 + 1, dtype=torch.float32) - radius
        kernel_1d = torch.exp(-(xs ** 2) / (2 * sigma ** 2))
        kernel_1d /= kernel_1d.sum()
        self.kernel_x = kernel_1d.view(1, 1, 1, -1).expand(3, 1, 1, -1).contiguous().to(self.device)
        self.kernel_y = kernel_1d.view(1, 1, -1, 1).expand(3, 1, -1, 1).contiguous().to(self.device)

    def blur(self, bgr_uint8):
        torch = self.torch
        F = torch.nn.functional
        t = torch.from_numpy(bgr_uint8).to(self.device, dtype=torch.float32)
        t = t.permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
        t = F.conv2d(t, self.kernel_x, padding=(0, self.pad), groups=3)
        t = F.conv2d(t, self.kernel_y, padding=(self.pad, 0), groups=3)
        t = t.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte()
        return t.cpu().numpy()


def has_audio_stream(path):
    """ffprobe check -- some CCTV clips have no audio track at all, and
    requesting -c:a copy / -map ...:a against a source with none makes ffmpeg
    hard-fail instead of just producing a video-only file."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    return bool(r.stdout.strip())


def censor_clip(src_path, out_path, points, effect="black", sigma=DEFAULT_BLUR_SIGMA,
                gpu_blur=None, progress_cb=None):
    """Apply `effect` ("black"/"blur" fill, or "crop") over `points`' region in
    every frame of src_path, mux the original audio back in (if any), write to
    out_path via ffmpeg. `gpu_blur`, if given, is a pre-built TorchGaussianBlur
    used instead of cv2.GaussianBlur for effect="blur".
    progress_cb(frames_done, frames_total) is called periodically if given."""
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        return False, "could not open video file"

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    if effect == "crop":
        # `points`' bounding box is the keep-box here (not a mask to hide --
        # the inverse of black/blur's meaning), so pick a separate region.json
        # via --region for this: reuses select_region.py's picker unmodified,
        # just clicking corners of the area to keep instead of the area to
        # hide. No masking at all: a smaller frame to encode beats even black
        # fill's per-pixel op, since libx264 cost scales with pixel count.
        # `points` is a list of shapes (see CONTEXT.md); with more than one,
        # the keep-box is the bounding box across all of them combined.
        all_pts = [p for shape in points for p in shape]
        crop_x0 = max(0, min(int(p[0]) for p in all_pts))
        crop_x1 = min(w, max(int(p[0]) for p in all_pts))
        crop_y0 = max(0, min(int(p[1]) for p in all_pts))
        crop_y1 = min(h, max(int(p[1]) for p in all_pts))
        out_w, out_h = max(1, crop_x1 - crop_x0), max(1, crop_y1 - crop_y0)
        mask_bool = mask_roi = None
    else:
        mask_bool = region_lib.region_mask((h, w), points, feather=0) > 0
        ys, xs = np.where(mask_bool)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        mask_roi = mask_bool[y0:y1, x0:x1]
        out_w, out_h = w, h

    audio_in = str(src_path) if has_audio_stream(src_path) else None
    cmd = ["ffmpeg", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{out_w}x{out_h}", "-r", str(fps),
           "-i", "-"]
    if audio_in:
        cmd += ["-i", audio_in, "-map", "0:v:0", "-map", "1:a:0", "-c:a", "copy"]
    else:
        cmd += ["-map", "0:v:0"]
    cmd += ["-c:v", "libx264", "-preset", PRESET, "-crf", str(CRF),
            "-movflags", "+faststart", str(out_path)]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    # Drain stderr on a separate thread while we write frames to stdin below --
    # ffmpeg's stderr pipe has a small OS buffer (~64KB); if it fills while we're
    # still writing frames (before stdin is closed), ffmpeg blocks on its own
    # stderr write, which blocks it from reading more stdin, which blocks our
    # writer -- a classic two-pipe deadlock. Reading stderr concurrently avoids it.
    stderr_chunks = []
    stderr_thread = threading.Thread(target=lambda: stderr_chunks.append(proc.stderr.read()))
    stderr_thread.start()
    write_error = None
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if effect == "crop":
                frame = frame[crop_y0:crop_y1, crop_x0:crop_x1]
            elif effect == "blur":
                roi = frame[y0:y1, x0:x1]
                blurred_roi = gpu_blur.blur(roi) if gpu_blur is not None else cv2.GaussianBlur(roi, (0, 0), sigma)
                roi[mask_roi] = blurred_roi[mask_roi]
            else:
                frame[mask_bool] = 0
            try:
                proc.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError) as e:
                # ffmpeg already exited (bad args for this clip, crashed, etc.) --
                # stop feeding it and report ffmpeg's own stderr below instead of
                # letting the raw pipe error escape and kill this worker's task.
                write_error = str(e)
                break
            frame_idx += 1
            if progress_cb and frame_idx % PROGRESS_EVERY_N_FRAMES == 0:
                progress_cb(frame_idx, total_frames)
    finally:
        cap.release()
        try:
            proc.stdin.close()
        except OSError:
            pass
        stderr_thread.join()
        stderr = stderr_chunks[0].decode(errors="replace") if stderr_chunks else ""
        proc.wait()
    if progress_cb:
        progress_cb(frame_idx, total_frames or frame_idx)

    if proc.returncode != 0 or write_error:
        reason = stderr[-500:] or write_error or "ffmpeg error"
        return False, reason
    return True, ""


def _censor_one(src_path, out_dir, points, force, effect, sigma, gpu_blur, progress_cb=None):
    """Never lets an exception escape -- one bad clip (corrupted file, weird
    resolution, disk hiccup, whatever) must not crash the whole batch."""
    out_path = out_dir / src_path.name
    if out_path.exists() and not force:
        return out_path.name, None, "already exists, skipped (--force to redo)"
    try:
        ok, reason = censor_clip(src_path, out_path, points, effect=effect, sigma=sigma,
                                 gpu_blur=gpu_blur, progress_cb=progress_cb)
    except Exception as e:
        ok, reason = False, f"{type(e).__name__}: {e}"
    return out_path.name, ok, reason


def _worker_loop(worker_idx, work_queue, out_dir, points, force, effect, sigma, use_gpu, progress_queue):
    """Persistent per-process loop: pull the next clip from the shared
    work_queue until it's empty, reporting progress through progress_queue
    (the only thing allowed to touch the terminal is the main process --
    multiple processes writing progress directly corrupts each other's
    output, same reasoning as motion_scan.py's --dynamic path).

    The GPU blur helper (if requested) is built once here, not per clip --
    a CUDA context is worth amortizing across everything this worker
    process handles (same reasoning as motion_scan.py's TorchMotionDetector)."""
    prefix = f"w{worker_idx}_"
    gpu_blur = TorchGaussianBlur(sigma) if (effect == "blur" and use_gpu and torch_cuda_available()) else None
    while True:
        try:
            src_path = work_queue.get_nowait()
        except queue_mod.Empty:
            break

        def cb(done, total, _prefix=prefix, _name=src_path.name):
            progress_queue.put(("progress", _prefix, _name, done, total))

        progress_queue.put(("progress", prefix, src_path.name, 0, None))
        result = _censor_one(src_path, out_dir, points, force, effect, sigma, gpu_blur, progress_cb=cb)
        progress_queue.put(("done", prefix, None, None, result))
    progress_queue.put(("worker_done", prefix, None, None, None))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", required=True,
                    help="Folder containing .mp4 clips directly (flat). Needs "
                         "region.json from select_region.py already saved here, "
                         "unless --region points elsewhere.")
    ap.add_argument("--region", default=None,
                    help="Path to region.json. Defaults to <folder>/region.json.")
    ap.add_argument("--workers", type=int, default=None,
                    help=f"Parallel clips to censor at once (default: os.cpu_count()-1 = "
                         f"{max(1, (os.cpu_count() or 2) - 1)})")
    ap.add_argument("--force", action="store_true",
                    help="Re-render clips that already exist in censored/ (default: skip them)")
    ap.add_argument("--effect", choices=["black", "blur", "crop"], default="black",
                    help="black: solid black fill (default -- see module docstring). "
                         "blur: gaussian blur over the region instead. "
                         "crop: cut every frame down to the region's bounding box (the box "
                         "to KEEP, not hide -- pick a separate region.json via --region for "
                         "this, e.g. select_region.py --region folder/region_crop.json) -- "
                         "smaller frame to encode, usually faster than black fill.")
    ap.add_argument("--blur-sigma", type=float, default=DEFAULT_BLUR_SIGMA,
                    help=f"Gaussian sigma for --effect blur (default: {DEFAULT_BLUR_SIGMA})")
    ap.add_argument("--gpu", action="store_true",
                    help="Use GPU (PyTorch CUDA) for --effect blur instead of CPU cv2.GaussianBlur. "
                         "No effect with --effect black. Falls back to CPU if no CUDA torch install found.")
    args = ap.parse_args()

    folder = Path(args.folder)
    region_path = Path(args.region) if args.region else folder / "region.json"
    if not region_path.is_file():
        sys.exit(f"{region_path} not found -- run select_region.py --folder \"{folder}\" first")

    region = region_lib.load_region(region_path)
    points = region["points"]

    clips = region_lib.discover_videos_flat(folder)
    out_dir = folder / "censored"
    out_dir.mkdir(exist_ok=True)

    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    workers = min(workers, len(clips)) or 1

    gpu_requested = args.gpu and args.effect == "blur"
    if gpu_requested and not torch_cuda_available():
        print("[censor] --gpu requested but no CUDA torch found -- falling back to CPU cv2.GaussianBlur")
    backend = "GPU (PyTorch CUDA)" if gpu_requested and torch_cuda_available() else "CPU"
    if args.effect == "blur":
        effect_desc = f"blur (sigma={args.blur_sigma}, {backend})"
    elif args.effect == "crop":
        effect_desc = "crop (region bounding box)"
    else:
        effect_desc = "black fill"
    print(f"found {len(clips)} files   workers={workers}   effect={effect_desc}   output={out_dir}")

    # spawn, not the platform default (fork on Linux/macOS) -- torch CUDA is
    # already touched in this process by torch_cuda_available() above, and a
    # forked child inheriting a CUDA-initialized parent hangs silently instead
    # of erroring when it tries to build its own context (confirmed in
    # practice: 6 workers stuck on futex_wait_queue indefinitely, 0% CPU, no
    # CUDA compute process registered in nvidia-smi). Windows only ever has
    # spawn, so this is a no-op there and keeps both platforms on one code path.
    mp_ctx = multiprocessing.get_context("spawn")
    manager = mp_ctx.Manager()
    work_queue = manager.Queue()
    for c in clips:
        work_queue.put(c)
    progress_queue = manager.Queue()

    idx_width = len(str(workers - 1)) if workers > 1 else 1

    overall_bar = tqdm(
        total=len(clips),
        unit=" clip",
        position=0,
        leave=True,
        bar_format="\033[1;96mOVERALL  | {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                   "[{elapsed}<{remaining}]\033[0m",
    )
    worker_bars = {
        f"w{i}_": tqdm(
            total=None,
            unit=" frames",
            position=i + 1,
            leave=True,
            bar_format="\033[97m{desc} | {n_fmt}/{total_fmt} frames [{elapsed}, {rate_fmt}]\033[0m",
        )
        for i in range(workers)
    }
    for i in range(workers):
        worker_bars[f"w{i}_"].set_description_str(f"w{i:0{idx_width}d}_idle")

    done = skipped = failed = 0
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as pool:
        futures = [pool.submit(_worker_loop, i, work_queue, out_dir, points, args.force,
                               args.effect, args.blur_sigma, gpu_requested, progress_queue)
                   for i in range(workers)]

        active_workers = workers
        while active_workers > 0:
            kind, prefix, name, done_frames, payload = progress_queue.get()
            bar = worker_bars[prefix]
            if kind == "worker_done":
                active_workers -= 1
                continue
            if kind == "progress":
                total = payload
                if total:
                    bar.total = total
                bar.n = done_frames
                bar.set_description_str(f"{prefix}{name}")
                bar.refresh()
                continue
            # kind == "done"
            out_name, ok, reason = payload
            if ok is None:
                skipped += 1
            elif ok:
                done += 1
            else:
                failed += 1
                tqdm.write(f"[FAIL] {out_name}: {reason}")
            overall_bar.update(1)
            bar.n = 0
            bar.total = None
            bar.set_description_str(f"{prefix}idle")
            bar.refresh()

        for fut in futures:
            fut.result()  # surface any worker-loop-level crash (shouldn't happen -- _censor_one
                          # catches per-clip exceptions -- but don't swallow it silently if it does)

    overall_bar.close()
    for bar in worker_bars.values():
        bar.close()

    print(f"finished  done {done}  skipped {skipped}  failed {failed}")


if __name__ == "__main__":
    main()
