#!/usr/bin/env python3
"""
detect_objects.py - Second-pass object-detection filter for motion_scan.py's
output clips: runs a pretrained YOLO model over the (already short,
motion-triggered) event clips and reports which ones actually contain a
class of interest (default: person, dog, cat), so you don't have to eyeball
every clip motion_scan.py produced.

Setup:
    pip install ultralytics

Usage:
    # Report which clips in <folder>/output contain person/dog/cat:
    python detect_objects.py --folder "F:/cam1"

    # Clips that match are copied into <folder>/detected/ for easy review
    # (on by default); pass --no-copy to skip that and just print the report:
    python detect_objects.py --folder "F:/cam1"

    # Custom classes (must be names the model knows -- COCO classes for the
    # default yolo11x.pt; run once and check the "unknown class" warning if
    # unsure what a model supports):
    python detect_objects.py --folder "F:/cam1" --classes dog,cat,bird

    # Run several worker processes at once (each with its own model
    # instance) to use GPU headroom a single stream leaves idle:
    python detect_objects.py --folder "F:/cam1" --workers 3
"""
import argparse
import concurrent.futures
import shutil
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".ts", ".dav"}

# When splitting work across --workers processes, only plan to use this
# fraction of currently-free VRAM -- leaves headroom for the desktop/other
# apps and for the calibration in _estimate_worker_vram_bytes() being an
# approximation (each worker also pays its own CUDA-context overhead, which
# a single calibration call inside the already-running main process can
# undercount slightly).
VRAM_SAFETY_MARGIN = 0.85


def discover_clips(folder):
    p = Path(folder)
    files = sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXTS)
    if not files:
        raise FileNotFoundError(f"No video clips found in {folder}")
    return files


def _run_batch(model, batch, conf, class_ids, found):
    for result in model.predict(batch, conf=conf, classes=class_ids, verbose=False):
        for box in result.boxes:
            found.add(int(box.cls))


def _transfer_clip(clip, detected_dir, move, no_copy):
    """Copies (or moves) a single matched clip into detected_dir right away,
    rather than waiting for every clip to finish scanning first -- so results
    are usable as they come in, and aren't lost if the run is interrupted
    partway through a long folder."""
    if no_copy:
        return
    detected_dir.mkdir(parents=True, exist_ok=True)
    dest = detected_dir / clip.name
    (shutil.move if move else shutil.copy2)(str(clip), str(dest))


def _scan_worker(model_path, clip_strs, class_ids, id_to_name, conf, frame_skip, batch_size,
                 detected_dir_str, move, no_copy):
    """Runs in its own process with its own model instance -- a loaded YOLO
    model holds a CUDA context, which can't be shared/pickled across
    processes, so each worker loads the model itself rather than the main
    process loading one and handing it out. Prints its own clips' results
    and transfers each match as it goes (matches motion_scan.py's --workers,
    where GPU headroom left unused by one stream gets used by running
    several concurrently)."""
    model = YOLO(model_path)
    detected_dir = Path(detected_dir_str)
    match_count = 0
    # Copying/moving runs on a background thread instead of blocking the
    # scan loop -- it's disk I/O, not GPU/CPU work, so there's no reason to
    # sit idle waiting on it before scanning the next clip. The `with` block
    # still waits for any still-in-flight transfers before this worker
    # reports done, so a match is never silently lost.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as copy_pool:
        for clip_str in clip_strs:
            clip = Path(clip_str)
            found_ids = scan_clip(model, clip, class_ids, conf, frame_skip, batch_size)
            if found_ids:
                found_names = sorted(id_to_name[i] for i in found_ids)
                print(f"MATCH  {clip.name}: {', '.join(found_names)}")
                copy_pool.submit(_transfer_clip, clip, detected_dir, move, no_copy)
                match_count += 1
            else:
                print(f"   --  {clip.name}")
    return match_count


def _estimate_worker_vram_bytes(model, sample_frame, batch_size, conf, class_ids):
    """One warm-up predict() call at the real --batch-size, to measure how
    much VRAM a single worker instance actually needs at that batch size --
    --workers and --batch-size both multiply GPU memory use, and a
    combination that doesn't fit can hang the whole machine (a CUDA OOM/driver
    reset on the card that's also driving the desktop), not just crash
    Python. Better to measure than guess a fixed number."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model.predict([sample_frame] * batch_size, conf=conf, classes=class_ids, verbose=False)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_reserved()


def scan_clip(model, path, class_ids, conf, frame_skip, batch_size):
    """Returns the set of matched class ids found in this clip. Samples every
    `frame_skip`th frame (clips are already short, and consecutive frames of
    the same event are near-duplicates), batching `batch_size` sampled frames
    per predict() call -- one frame per call leaves the GPU idle between
    calls waiting on the next frame's CPU decode, so batching keeps it fed
    with more work per call instead. Stops early once every requested class
    has been seen at least once (may run one batch past that point, since a
    match is only checked between batches, not mid-batch)."""
    cap = cv2.VideoCapture(str(path))
    found = set()
    idx = -1
    batch = []
    while len(found) < len(class_ids):
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % frame_skip != 0:
            continue
        batch.append(frame)
        if len(batch) >= batch_size:
            _run_batch(model, batch, conf, class_ids, found)
            batch = []
    if batch and len(found) < len(class_ids):
        _run_batch(model, batch, conf, class_ids, found)
    cap.release()
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", default=None,
                         help="Camera folder containing output/ (motion_scan.py's clips). "
                              "Defaults --input to <folder>/output.")
    parser.add_argument("--input", default=None, help="Folder of clips to scan. Defaults to <folder>/output.")
    parser.add_argument("--classes", default="person,dog,cat",
                         help="Comma-separated class names to look for. Default: person,dog,cat.")
    parser.add_argument("--conf", type=float, default=0.4, help="Minimum detection confidence. Default 0.4.")
    parser.add_argument("--frame-skip", type=int, default=5,
                         help="Only run detection every Nth frame of each clip. Default 5.")
    parser.add_argument("--batch-size", type=int, default=16,
                         help="Sampled frames sent to the model per predict() call. Higher keeps the "
                              "GPU busier (fewer, bigger calls) at the cost of more memory. Default 16.")
    parser.add_argument("--model", default="yolo11x.pt",
                         help="Ultralytics model to use (auto-downloaded if not already present). "
                              "Default yolo11x.pt.")
    parser.add_argument("--no-copy", action="store_true",
                         help="Don't copy matched clips anywhere -- just print the report.")
    parser.add_argument("--move", action="store_true",
                         help="Move matched clips instead of copying them (removes them from the input folder).")
    parser.add_argument("--workers", type=int, default=5,
                         help="Split clips across N worker processes, each with its own model instance. "
                              "A single stream often can't saturate the GPU on its own -- running several "
                              "at once uses headroom that would otherwise go to waste. Auto-reduced if it "
                              "wouldn't fit in free VRAM. Default 5.")
    args = parser.parse_args()

    input_dir = args.input or (str(Path(args.folder) / "output") if args.folder else None)
    if not input_dir:
        raise SystemExit("Need --input, or --folder (which defaults --input to <folder>/output).")

    # Default destination is <folder>/detected (a sibling of output/, so it's
    # easy to find without digging into output/). Without --folder there's no
    # "camera folder" to put it next to, so fall back to <input>/detected.
    detected_dir = Path(args.folder) / "detected" if args.folder else Path(input_dir) / "detected"

    requested = [c.strip() for c in args.classes.split(",") if c.strip()]
    clips = discover_clips(input_dir)
    model = YOLO(args.model)  # loaded once here just to resolve class names -> ids below

    name_to_id = {name: idx for idx, name in model.names.items()}
    class_ids = [name_to_id[c] for c in requested if c in name_to_id]
    unknown = [c for c in requested if c not in name_to_id]
    if unknown:
        print(f"[detect-objects] Ignoring class(es) this model doesn't know: {', '.join(unknown)}")
    if not class_ids:
        raise SystemExit("None of the requested classes are known to this model -- nothing to detect.")

    id_to_name = {i: model.names[i] for i in class_ids}
    workers = max(1, min(args.workers, len(clips)))

    if workers == 1:
        match_count = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as copy_pool:
            for clip in clips:
                found_ids = scan_clip(model, clip, class_ids, args.conf, args.frame_skip, args.batch_size)
                if found_ids:
                    found_names = sorted(id_to_name[i] for i in found_ids)
                    print(f"MATCH  {clip.name}: {', '.join(found_names)}")
                    copy_pool.submit(_transfer_clip, clip, detected_dir, args.move, args.no_copy)
                    match_count += 1
                else:
                    print(f"   --  {clip.name}")
    else:
        if torch.cuda.is_available():
            sample_frame = cv2.VideoCapture(str(clips[0])).read()[1]
            per_worker_bytes = _estimate_worker_vram_bytes(model, sample_frame, args.batch_size,
                                                           args.conf, class_ids)
            del model  # each worker below loads its own instance; free this one's GPU memory first
            torch.cuda.empty_cache()
            free_bytes, _total_bytes = torch.cuda.mem_get_info()
            usable_bytes = int(free_bytes * VRAM_SAFETY_MARGIN)
            max_by_vram = max(1, int(usable_bytes // per_worker_bytes))
            if workers > max_by_vram:
                print(f"[detect-objects] --workers {workers} at --batch-size {args.batch_size} would need "
                      f"~{workers * per_worker_bytes / 2**30:.1f}GB VRAM, but only "
                      f"~{free_bytes / 2**30:.1f}GB is free -- capping to --workers {max_by_vram} "
                      "so this doesn't hang the machine. Lower --batch-size to raise this ceiling.")
                workers = max_by_vram
        else:
            del model  # each worker below loads its own instance; no VRAM to check without CUDA

        # Round-robin, not contiguous chunks: clips are sorted by timestamp, and
        # a burst of consecutive clips can be similarly short/long (e.g. a
        # quiet vs. busy stretch of the day), so interleaving spreads that
        # unevenness across workers instead of handing one worker a whole
        # uneven run.
        chunks = [[str(c) for c in clips[i::workers]] for i in range(workers)]
        chunks = [c for c in chunks if c]
        match_count = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=len(chunks)) as pool:
            futures = [pool.submit(_scan_worker, args.model, chunk, class_ids, id_to_name,
                                    args.conf, args.frame_skip, args.batch_size,
                                    str(detected_dir), args.move, args.no_copy)
                       for chunk in chunks]
            for fut in concurrent.futures.as_completed(futures):
                match_count += fut.result()

    print(f"\n{match_count}/{len(clips)} clip(s) matched {sorted(id_to_name.values())}.")
    if not args.no_copy and match_count:
        verb = "Moved" if args.move else "Copied"
        print(f"{verb} each match to {detected_dir} as it was found.")


if __name__ == "__main__":
    main()
