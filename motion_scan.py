#!/usr/bin/env python3
"""
motion_scan.py - Motion detection over a user-picked region, across long
sequences of chained DVR camera files (e.g. Hikvision exports split into
many .mp4 segments for one day).

Runs CPU background subtraction (OpenCV MOG2) by default. GPU compute
(PyTorch + CUDA background subtraction, `--gpu`) and GPU video decode
(ffmpeg NVDEC, `--nvdec`) both exist but are OFF by default: on this
project's hardware CPU compute measured faster at every worker count
tested (2026-07-06: 638.9 vs 152.3fps at --workers 10), and the GPU
detector's fixed diff>4*std threshold misses low-contrast night/IR motion
that MOG2 catches -- see DESIGN.md/PROGRESS.md before enabling either.

GPU SETUP (only if you want to re-test `--gpu`/`--nvdec` on your own
hardware; no custom OpenCV build needed):
    1. Install a CUDA build of PyTorch -- pick the command for your CUDA
       version at https://pytorch.org/get-started/locally/, e.g.:
           pip install torch --index-url https://download.pytorch.org/whl/cu124
    2. (optional, for GPU video decode too) Install an ffmpeg build with
       NVDEC/cuvid support (most full Windows builds, e.g. gyan.dev "full"
       builds, already have it) and make sure `ffmpeg`/`ffprobe` are on PATH.
    The console output on each run prints which backend is actually active.

Usage:
    Each camera/location gets its own self-contained folder:
        <folder>/data/         source videos go here
        <folder>/output/       event clips get written here
        <folder>/region.json   detection zone for this folder, created by select-region

    # 1. Pick the detection zone once per camera (click points, right-click
    #    to undo last point, Enter/S to save, Esc/Q to cancel):
    python motion_scan.py select-region --folder "F:/cam1"

    # 2. Scan all files in <folder>/data using the saved region, write clips to <folder>/output:
    python motion_scan.py scan --folder "F:/cam1"

    # --input/--region/--output can still be passed explicitly to override any
    # of the --folder defaults, e.g. to reuse one region.json across folders:
    python motion_scan.py scan --folder "F:/cam1" --region "F:/shared_region.json"
"""

import argparse
import hashlib
import json
import os
import queue as queue_mod
import re
import subprocess
from collections import deque
from pathlib import Path

# Must be set before cv2's ffmpeg backend initializes (it reads this once,
# lazily, on first use) -- silences noisy decoder warnings like "Could not
# find ref with POC ..." that ffmpeg prints straight to the process's
# stderr handle. They happen when seeking into the middle of a GOP for a
# segment start (see SEEK_WARMUP_FRAMES / ChainedVideoReader._open_next
# below -- that same seek was also found to occasionally produce visibly
# corrupted frames, not just this log noise). Silenced here regardless
# because they bypass Python entirely and corrupt the multi-worker progress
# bar layout by printing in between tqdm's cursor-position escape codes.
# -8 = AV_LOG_QUIET (errors still surface via OpenCV's own return codes,
# just not ffmpeg's raw log text).
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

import cv2
import numpy as np
from tqdm import tqdm

import select_region as region_lib

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".ts", ".dav"}

# Frames to decode-and-discard before a segment's real start after seeking
# into it (see the comment in ChainedVideoReader._open_next). 2s at a typical
# 25fps camera -- comfortably more than one GOP for camera footage, so the
# decoder has resynced its reference chain well before real frames begin.
SEEK_WARMUP_FRAMES = 50

# Target frames per segment when cutting a file up for --workers balancing
# (see scan()) -- fixed at a flat frame count, not a time duration,
# deliberately ignoring fps, so the same number always means the same
# segment size regardless of what footage it's run on. Also used by
# _scan_dynamic() as the floor for its own (larger) dispatch piece size.
TARGET_SEGMENT_FRAMES = 1000


# --------------------------------------------------------------------------
# File discovery / ordering
# --------------------------------------------------------------------------

def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def discover_videos(input_path):
    p = Path(input_path)
    if p.is_file():
        return [p]
    files = [f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXTS]
    files.sort(key=lambda f: natural_key(f.name))
    if not files:
        raise FileNotFoundError(f"No video files found in {input_path}")
    return files


# --------------------------------------------------------------------------
# Chained reader: many files -> one continuous frame stream
# --------------------------------------------------------------------------

class ChainedVideoReader:
    """Presents a sorted list of video files as one continuous frame stream."""

    def __init__(self, paths):
        # Each item is either a bare Path (whole file) or a 5-tuple
        # (path, start_frame, end_frame, part_idx, total_parts) describing a
        # sub-range of frames within that file. The tuple form lets a single
        # large file be split into several independently-readable pieces,
        # e.g. for finer-grained load balancing across worker processes; a
        # file that wasn't split has total_parts == 1.
        self.segments = [item if isinstance(item, tuple) else (item, 0, None, 1, 1) for item in paths]
        self.paths = [s[0] for s in self.segments]
        self.fps = None
        self.width = None
        self.height = None
        self.file_frame_counts = []
        self._probe()
        self._file_idx = -1
        self._cap = None
        self._seg_remaining = 0
        self.global_index = -1

    def _probe(self):
        meta_cache = {}
        for p, start, end, _part_idx, _total_parts in self.segments:
            key = str(p)
            if key not in meta_cache:
                cap = cv2.VideoCapture(key)
                if not cap.isOpened():
                    raise IOError(f"Cannot open {p}")
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                meta_cache[key] = (fps, w, h, count)
            fps, w, h, count = meta_cache[key]
            if self.fps is None:
                self.fps, self.width, self.height = fps, w, h
            elif (w, h) != (self.width, self.height):
                raise ValueError(
                    f"{p.name} is {w}x{h}, but {self.paths[0].name} is "
                    f"{self.width}x{self.height}. All files in one scan must "
                    "be from the same camera/resolution."
                )
            seg_end = count if end is None else min(end, count)
            self.file_frame_counts.append(max(seg_end - start, 0))
        self.total_frames = sum(self.file_frame_counts)

    def _open_next(self):
        self._file_idx += 1
        if self._cap is not None:
            self._cap.release()
        if self._file_idx >= len(self.segments):
            self._cap = None
            return False
        path, start, _end, _part_idx, _total_parts = self.segments[self._file_idx]
        self._cap = cv2.VideoCapture(str(path))
        if start:
            # cv2.CAP_PROP_POS_FRAMES seeks to the nearest position ffmpeg's
            # demuxer can reach directly, then decodes forward from there --
            # for GOP-coded formats (HEVC/H.264) that can land mid-GOP without
            # every reference frame the decoder needs, producing a visibly
            # corrupted frame (partial content over a mostly-gray fill) for a
            # moment right after the seek, before the reference chain
            # recovers. This previously showed up as corrupted clips whenever
            # --workers split a file into segments (every non-first segment
            # seeks to start its own mid-file offset). Seeking a bit earlier
            # and decoding-forward past SEEK_WARMUP_FRAMES, discarding those
            # frames, gives the decoder room to resync before `start` --
            # the actual segment content -- is reached.
            warmup_start = max(start - SEEK_WARMUP_FRAMES, 0)
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, warmup_start)
            for _ in range(start - warmup_start):
                if not self._cap.read()[0]:
                    break
        self._seg_remaining = self.file_frame_counts[self._file_idx]
        return self._cap.isOpened()

    def read(self):
        """Returns (ok, frame); transparently advances to the next file/segment
        once the current one is exhausted."""
        while True:
            if self._cap is None or self._seg_remaining <= 0:
                if not self._open_next():
                    return False, None
                continue
            ok, frame = self._cap.read()
            if not ok:
                # Physical EOF reached before the expected segment length
                # (e.g. a slightly-off frame count from the container) --
                # just move on to the next segment/file.
                self._seg_remaining = 0
                continue
            self._seg_remaining -= 1
            self.global_index += 1
            return True, frame

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def first_frame(self):
        """Peek the first frame of the first file without touching playback state."""
        cap = cv2.VideoCapture(str(self.paths[0]))
        ok, frame = cap.read()
        cap.release()
        return frame if ok else None

    def filename_for_index(self, global_idx):
        """Returns the name to use for an output clip starting at frame
        `global_idx`: the source file's stem, with a `_pN` suffix appended
        if that file was split into multiple segments (so two segments of
        the same original file never produce colliding clip names). The
        part number is zero-padded to the width of the largest part number
        for that file, so e.g. p2 sorts before p10 -- without padding,
        plain string sort (which is what a file browser/`ls` uses) puts
        "_p10" before "_p2", scrambling playback order for any file split
        into 10+ parts."""
        cum = 0
        for (p, _start, _end, part_idx, total_parts), count in zip(self.segments, self.file_frame_counts):
            cum += count
            if global_idx < cum:
                return self._part_name(p, part_idx, total_parts)
        if self.segments:
            p, _start, _end, part_idx, total_parts = self.segments[-1]
            return self._part_name(p, part_idx, total_parts)
        return "unknown"

    @staticmethod
    def _part_name(p, part_idx, total_parts):
        if total_parts <= 1:
            return p.stem
        width = len(str(total_parts))
        return f"{p.stem}_p{part_idx:0{width}d}"

    def segments_for_range(self, start_idx, end_idx):
        """Returns the sub-list of (path, start, end, part_idx, total_parts)
        segments covering this reader's piece-global frames [start_idx,
        end_idx) exactly, clamped against self.segments/file_frame_counts
        (an event can span more than one segment/file if it crosses a
        boundary within this piece). Used by EventRecorder's manifest-only
        mode (see below) to resolve an event's frame range into absolute
        per-file positions that the `extract` command can re-read later
        without re-running detection -- the same seek-with-warmup path a
        normal ChainedVideoReader already uses, just fed a smaller slice."""
        result = []
        cum = 0
        for (p, seg_start, _seg_end, part_idx, total_parts), count in zip(self.segments, self.file_frame_counts):
            seg_lo, seg_hi = cum, cum + count
            cum = seg_hi
            lo, hi = max(start_idx, seg_lo), min(end_idx, seg_hi)
            if lo >= hi:
                continue
            abs_start = seg_start + (lo - seg_lo)
            abs_end = seg_start + (hi - seg_lo)
            result.append((p, abs_start, abs_end, part_idx, total_parts))
        return result


# --------------------------------------------------------------------------
# Region selection UI -- picker UI + region.json format live in
# select_region.py (shared with censor.py); this just supplies the preview
# frame the way this tool's <folder>/data layout requires.
# --------------------------------------------------------------------------

def select_region(input_path, region_path):
    paths = discover_videos(input_path)
    reader = ChainedVideoReader(paths)
    frame = reader.first_frame()
    reader.release()
    if frame is None:
        raise RuntimeError("Could not read a frame to select a region from.")
    region_lib.select_region(input_path, region_path, frame)


# --------------------------------------------------------------------------
# CPU motion detector: ROI-masked MOG2 (always available, no extra installs)
# --------------------------------------------------------------------------

class MotionDetector:
    """ROI-masked CPU background subtraction (OpenCV MOG2). Crops to the
    region's bounding box before running MOG2, so cost scales with the
    selected zone, not the whole frame. This is the fallback used when GPU
    compute (see TorchMotionDetector) isn't available."""

    def __init__(self, frame_w, frame_h, points, var_threshold=16.0, downscale=1):
        self.downscale = max(int(downscale), 1)

        # `points` is a list of shapes (see CONTEXT.md); the bounding box
        # spans all of them combined, and each shape gets fillPoly'd into
        # the same mask -- multiple shapes are one merged region, not
        # separate per-shape masks.
        all_pts = np.array([pt for shape in points for pt in shape], dtype=np.int32)
        x, y, w, h = cv2.boundingRect(all_pts)
        x, y = max(x, 0), max(y, 0)
        w, h = min(w, frame_w - x), min(h, frame_h - y)
        if w <= 0 or h <= 0:
            raise ValueError("Region points produce an empty bounding box.")
        self.roi_rect = (x, y, w, h)

        local_polys = [np.array(shape, dtype=np.int32) - [x, y] for shape in points]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, local_polys, 255)
        if self.downscale > 1:
            mask = cv2.resize(mask, (max(w // self.downscale, 1), max(h // self.downscale, 1)),
                              interpolation=cv2.INTER_NEAREST)
        self.mask = mask
        self.roi_area = int(cv2.countNonZero(self.mask))
        self._kernel = np.ones((3, 3), np.uint8)

        self.bg = cv2.createBackgroundSubtractorMOG2()
        self.bg.setVarThreshold(var_threshold)
        try:
            self.bg.setDetectShadows(False)
        except Exception:
            pass

    def score(self, frame):
        """Returns the fraction (0-1) of ROI pixels flagged as foreground."""
        x, y, w, h = self.roi_rect
        crop = frame[y:y + h, x:x + w]
        if self.downscale > 1:
            crop = cv2.resize(crop, (self.mask.shape[1], self.mask.shape[0]), interpolation=cv2.INTER_AREA)

        fg = self.bg.apply(crop)
        fg = cv2.bitwise_and(fg, fg, mask=self.mask)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self._kernel)
        return cv2.countNonZero(fg) / max(self.roi_area, 1)


# --------------------------------------------------------------------------
# GPU motion detector: PyTorch CUDA, adaptive single-Gaussian background
# --------------------------------------------------------------------------

def torch_cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


class TorchMotionDetector:
    """GPU-accelerated background subtraction using a per-pixel adaptive
    mean/variance model (EMA), masked to the selected region, run on CUDA
    via PyTorch.

    This is the supported way to get real GPU acceleration here: a plain
    `pip install torch` with a CUDA build (see pytorch.org) is enough -- no
    custom OpenCV-with-CUDA build required.
    """

    def __init__(self, frame_w, frame_h, points, alpha=0.02, k=4.0, min_std=6.0):
        import torch
        self.torch = torch
        self.device = torch.device("cuda")

        # `points` is a list of shapes (see CONTEXT.md); the bounding box
        # spans all of them combined, and each shape gets fillPoly'd into
        # the same mask -- multiple shapes are one merged region, not
        # separate per-shape masks.
        all_pts = np.array([pt for shape in points for pt in shape], dtype=np.int32)
        x, y, w, h = cv2.boundingRect(all_pts)
        x, y = max(x, 0), max(y, 0)
        w, h = min(w, frame_w - x), min(h, frame_h - y)
        if w <= 0 or h <= 0:
            raise ValueError("Region points produce an empty bounding box.")
        self.roi_rect = (x, y, w, h)

        local_polys = [np.array(shape, dtype=np.int32) - [x, y] for shape in points]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, local_polys, 1)
        self.mask = torch.from_numpy(mask.astype(np.float32)).to(self.device)
        self.roi_area = float(mask.sum())

        self.alpha = alpha
        self.k = k
        self.min_var = min_std ** 2
        self.bg_mean = None
        self.bg_var = None

    def score(self, frame):
        """Returns the fraction (0-1) of ROI pixels flagged as foreground."""
        torch = self.torch
        x, y, w, h = self.roi_rect
        crop = frame[y:y + h, x:x + w]
        gray = torch.from_numpy(crop).to(self.device, dtype=torch.float32).mean(dim=2)

        if self.bg_mean is None:
            self.bg_mean = gray.clone()
            self.bg_var = torch.full_like(gray, self.min_var)
            return 0.0

        diff = (gray - self.bg_mean).abs()
        std = self.bg_var.clamp(min=self.min_var).sqrt()
        fg = (diff > self.k * std).float() * self.mask
        score = (fg.sum() / self.roi_area).item()

        # Only adapt the background where this frame did NOT look like motion,
        # so a slow-moving subject doesn't get absorbed into the background.
        update = (diff <= self.k * std).float()
        self.bg_mean = self.bg_mean + self.alpha * update * (gray - self.bg_mean)
        self.bg_var = self.bg_var + self.alpha * update * ((gray - self.bg_mean) ** 2 - self.bg_var)
        return score


# --------------------------------------------------------------------------
# Optional NVDEC decode: ffmpeg hardware decode -> raw BGR frames
# --------------------------------------------------------------------------

CUVID_DECODERS = {
    "h264": "h264_cuvid",
    "hevc": "hevc_cuvid",
    "mpeg4": "mpeg4_cuvid",
    "mjpeg": "mjpeg_cuvid",
    "vp9": "vp9_cuvid",
    "vp8": "vp8_cuvid",
}


def probe_codec(path):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


class FFmpegNVDECReader:
    """Decodes a chain of video files using ffmpeg's NVDEC hardware decoder,
    yielding raw BGR frames -- this is what offloads video decoding (not
    just motion detection) onto the GPU. Requires an ffmpeg build with CUDA
    decoder support (most full Windows builds, e.g. from gyan.dev, have it).

    Construction raises RuntimeError if NVDEC decode can't be confirmed; the
    caller should catch this and fall back to CPU decode (ChainedVideoReader).

    Accepts the same items as ChainedVideoReader: a bare Path (whole file)
    or a (path, start, end, part_idx, total_parts) segment tuple. This
    matters because `--workers > 1` always hands pieces built from segment
    tuples (see TARGET_SEGMENT_FRAMES in scan()), never bare paths -- an
    earlier version of this class only handled bare paths, so any
    `--workers > 1` run combined with `--nvdec` was stringifying a raw
    Python tuple into the `-i` argument, ffmpeg failed to open it
    immediately, and every worker silently fell back to CPU decode with no
    indication why (found 2026-07-06 while testing `--nvdec-workers`; the
    original "NVDEC is slower" benchmark in CLAUDE.md/DESIGN.md was run at
    `--workers 1`, the one case this bug can't fire in, so it went
    unnoticed). A segment's `start` offset is honored the same way
    ChainedVideoReader does it -- seek `SEEK_WARMUP_FRAMES` early via `-ss`
    and decode-forward, discarding frames, before the segment's real
    content -- for the same reason: landing exactly on `start` via a
    from-scratch seek can start mid-GOP and hand back corrupted frames
    before the decoder's reference chain resyncs. Unlike that path, this
    one hasn't been verified frame-accurate against ground truth -- treat
    it as experimental.
    """

    def __init__(self, paths, width, height, fps, ffmpeg_bin="ffmpeg"):
        self.segments = [item if isinstance(item, tuple) else (item, 0, None, 1, 1) for item in paths]
        self.width = width
        self.height = height
        self.fps = fps or 25.0
        self.frame_bytes = width * height * 3
        self.ffmpeg_bin = ffmpeg_bin
        self._file_idx = -1
        self._proc = None
        self._remaining = None  # frames left in the current segment; None = whole file, no limit
        self._codec_cache = {}  # path (str) -> probed codec name, since a file's segments share one codec
        self.global_index = -1

        if not self._open_next():
            raise RuntimeError("could not start ffmpeg NVDEC decode")
        ok, _ = self.read()
        if not ok:
            raise RuntimeError("ffmpeg NVDEC decode produced no frames")
        # restart clean so the caller gets every frame from the beginning
        self.release()
        self._file_idx = -1
        self.global_index = -1
        self._open_next()

    def _open_next(self):
        self._file_idx += 1
        if self._proc is not None:
            self._proc.stdout.close()
            self._proc.wait(timeout=5)
        if self._file_idx >= len(self.segments):
            self._proc = None
            return False
        path, start, end, _part_idx, _total_parts = self.segments[self._file_idx]
        key = str(path)
        if key not in self._codec_cache:
            self._codec_cache[key] = probe_codec(path) or "h264"
        decoder = CUVID_DECODERS.get(self._codec_cache[key], "h264_cuvid")
        cmd = [self.ffmpeg_bin, "-hwaccel", "cuda", "-c:v", decoder]
        warmup = 0
        if start:
            warmup_start = max(start - SEEK_WARMUP_FRAMES, 0)
            cmd += ["-ss", f"{warmup_start / self.fps:.6f}"]
            warmup = start - warmup_start
        cmd += ["-i", str(path), "-an", "-pix_fmt", "bgr24", "-f", "rawvideo"]
        if end is not None:
            cmd += ["-frames:v", str(warmup + (end - start))]
        cmd += ["-"]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                      bufsize=self.frame_bytes * 4)
        for _ in range(warmup):
            if not self._proc.stdout.read(self.frame_bytes):
                break
        self._remaining = None if end is None else (end - start)
        return True

    def read(self):
        if self._proc is None:
            if not self._open_next():
                return False, None
        if self._remaining is not None and self._remaining <= 0:
            return self.read() if self._open_next() else (False, None)
        buf = self._proc.stdout.read(self.frame_bytes)
        if len(buf) < self.frame_bytes:
            return self.read() if self._open_next() else (False, None)
        frame = np.frombuffer(buf, dtype=np.uint8).reshape(self.height, self.width, 3)
        self.global_index += 1
        if self._remaining is not None:
            self._remaining -= 1
        return True, frame

    def release(self):
        if self._proc is not None:
            try:
                self._proc.stdout.close()
                self._proc.terminate()
            except Exception:
                pass
            self._proc = None


# --------------------------------------------------------------------------
# Event state machine -> clip files
# --------------------------------------------------------------------------

class EventRecorder:
    def __init__(self, output_dir, fps, frame_size, pre_frames, post_frames, codec="mp4v", source_name_fn=None,
                 manifest_only=False, range_fn=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.frame_size = frame_size
        self.pre_frames = pre_frames
        self.post_frames = post_frames
        self.codec = codec
        self.source_name_fn = source_name_fn
        # manifest_only: skip writing clip video files entirely and instead
        # log each event's source file(s)/frame-range to self.manifest_events
        # (see scan()'s --manifest-only). range_fn resolves an event's
        # piece-global [start, end) into absolute per-file segments --
        # normally ChainedVideoReader.segments_for_range. Built for the
        # cloud pilot: scanning on a rented VM and downloading real clips
        # back costs real egress money (GCP bills ~$0.09-0.12/GB out,
        # ingress is free), while a manifest of (file, start, end) triples
        # is a few KB regardless of how much footage was scanned -- the
        # actual clips get cut afterward on the local machine (which still
        # has the source footage) via the `extract` command, at zero
        # detection cost, using the exact same segment-seek path as a
        # normal scan.
        self.manifest_only = manifest_only
        self.range_fn = range_fn
        self.manifest_events = []
        self.pre_buffer = deque(maxlen=pre_frames)
        self.writer = None
        self._in_event = False
        self._current_name = None
        self._current_start_idx = None
        self.frames_since_motion = 0
        self.event_index = 0  # total events found by this worker (for progress display only)
        # Per-original-file event counters, used for the clip filename suffix.
        # A single worker's chunk can contain segments from several different
        # source files (contiguous load balancing doesn't keep one file per
        # worker -- a worker's chunk can start partway through one file and
        # end partway through another), so a single running counter would
        # number a file's events based on
        # what else the worker happened to process before it -- e.g. file B's
        # first event could be named "..._0007" just because the worker found
        # 6 events in file A first. Counting per file-name instead means each
        # file's clips are always "..._0001", "..._0002", ... in order,
        # regardless of what else shares the worker.
        self._file_event_counts = {}

    def in_event(self):
        return self._in_event

    def push_frame(self, frame, global_idx, motion_now, confirmed_motion):
        if self.in_event():
            if self.writer is not None:
                self.writer.write(frame)
            if motion_now:
                self.frames_since_motion = 0
            else:
                self.frames_since_motion += 1
            if self.frames_since_motion >= self.post_frames:
                self._close_event(global_idx)
        else:
            # manifest_only never needs the actual pixels, just the count of
            # how many frames are buffered -- storing `True` instead of the
            # frame avoids holding pre_frames full-resolution images in
            # memory for a mode that will never write them anywhere.
            self.pre_buffer.append(frame if not self.manifest_only else True)
            if confirmed_motion:
                self._open_event(global_idx)

    def _open_event(self, global_idx):
        self.event_index += 1
        event_start_idx = global_idx - len(self.pre_buffer) + 1
        original_name = self.source_name_fn(event_start_idx) if self.source_name_fn else "clip"
        file_event_idx = self._file_event_counts.get(original_name, 0) + 1
        self._file_event_counts[original_name] = file_event_idx
        name = f"{original_name}_{file_event_idx:04d}.mp4"
        self._current_name = name
        self._current_start_idx = event_start_idx
        if not self.manifest_only:
            path = self.output_dir / name
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.writer = cv2.VideoWriter(str(path), fourcc, self.fps, self.frame_size)
            for f in self.pre_buffer:
                self.writer.write(f)
        self.pre_buffer.clear()
        self.frames_since_motion = 0
        self._in_event = True

    def _close_event(self, end_idx):
        if self.manifest_only:
            segments = self.range_fn(self._current_start_idx, end_idx + 1) if self.range_fn else []
            self.manifest_events.append({
                "clip_name": self._current_name,
                "segments": [[p.name, start, end] for (p, start, end, _part_idx, _total_parts) in segments],
            })
        else:
            self.writer.release()
            self.writer = None
        self._in_event = False

    def finalize(self, end_idx=None):
        if self.in_event():
            self._close_event(end_idx)


# --------------------------------------------------------------------------
# Resume / save-state support
# --------------------------------------------------------------------------
#
# Granularity: one "unit" of resumable progress is one fixed-size segment
# (see TARGET_SEGMENT_FRAMES in scan()), identified by (file path, start
# frame, end frame) -- not the whole worker chunk it happens to land in. As
# soon as a worker finishes reading through one of its segments, it reports
# that segment as done and the main process persists it to the state file
# immediately, instead of waiting for the worker's entire chunk (which can
# be many segments) to finish. On a crash/hang, the next run skips every
# segment already marked done and only re-does what's left.
#
# Simple safety margin: when resuming, the *last* segment that was marked
# done for a given file is re-done anyway (stepped back by one), even though
# it's already complete. This is cheap insurance against the one edge case
# segment-level tracking doesn't try to solve precisely -- an event that was
# still open (motion ongoing) right at the moment of the crash, spanning
# from that last "done" segment into the next one. Redoing it from scratch
# costs at most one extra segment of decode time, not a full chunk.
#
# Segment *composition* (the frame ranges each file is cut into) is fully
# deterministic given the same inputs, so a segment's (path, start, end) key
# alone identifies "the same segment" across runs -- as long as nothing
# about the job changed. The fingerprint below exists to catch the case
# where something *did* change (different files, edited region, different
# settings) so we don't skip segments based on stale, no-longer-matching
# progress.
#
# NOTE: with --workers 1 the job is never split into segments at all (see
# scan()), so resume there stays all-or-nothing: a crash partway through a
# single-worker run still has to redo that run from the top.

def _state_path(output_dir):
    return Path(output_dir) / ".motion_scan_state.json"


def _fingerprint_job(paths, region_path, output_dir, threshold, min_event_len,
                     pre, post, use_gpu, downscale, use_nvdec, frame_skip, manifest_only):
    """Hashes everything that determines *what work needs doing*, so a
    saved state file is only trusted if none of it changed since it was
    written (different/added/removed/edited input files, a different
    region, or different scan settings all invalidate it). --workers is
    deliberately NOT part of this: progress is tracked per-segment, and
    segments get re-bucketed across however many workers you ask for on
    every run, so changing --workers between runs (e.g. tuning it down
    after a first overambitious run) shouldn't throw away saved progress."""
    # Paths are hashed as relative, forward-slash (`as_posix()`) strings, and
    # `output` isn't hashed at all -- so a state file stays valid when resumed
    # from a different OS/machine than the one that wrote it (e.g. picking up
    # a Windows run on Linux), as long as the same relative --folder/--input
    # is used. `str(Path(...))` uses OS-native separators (backslash on
    # Windows) and `.resolve()` bakes in a machine-specific absolute path --
    # either one guarantees a hash mismatch across machines even when nothing
    # about the actual job changed. `output` is redundant to hash anyway:
    # state_path is already output_dir/.motion_scan_state.json, so loading a
    # given state file already implies that output_dir.
    file_sig = []
    for p in paths:
        st = p.stat()
        file_sig.append([p.as_posix(), st.st_size, int(st.st_mtime)])
    region_p = Path(region_path)
    region_sig = [region_p.as_posix(), int(region_p.stat().st_mtime)] if region_p.exists() else None
    payload = {
        "files": file_sig,
        "region": region_sig,
        "threshold": threshold,
        "min_event_len": min_event_len,
        "pre": pre,
        "post": post,
        "use_gpu": use_gpu,
        "downscale": downscale,
        "use_nvdec": use_nvdec,
        "frame_skip": frame_skip,
        "manifest_only": manifest_only,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _load_state(state_path, fingerprint):
    """Returns the saved state dict if it exists and matches `fingerprint`,
    else None (no usable saved progress)."""
    if not state_path.exists():
        return None
    try:
        with open(state_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("fingerprint") != fingerprint:
        print(f"[motion-scan] Found a previous progress file at {state_path}, but the job has "
              "changed (different files, region, or settings) -- ignoring it and starting fresh.")
        return None
    return data


def _save_state(state_path, fingerprint, **fields):
    """Writes atomically (temp file + rename) so a crash mid-write never
    leaves a corrupt/half-written state file behind."""
    data = {"fingerprint": fingerprint, **fields}
    tmp_path = state_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f)
    os.replace(tmp_path, state_path)


def _clear_state(state_path):
    try:
        state_path.unlink()
    except FileNotFoundError:
        pass


# --------------------------------------------------------------------------
# Main scan
# --------------------------------------------------------------------------

def _process_piece(piece_paths, points, output_dir, threshold, min_event_len, pre, post,
                    use_gpu, downscale, use_nvdec, frame_skip, name_prefix, progress_queue,
                    frames_done, total_events, pbar, manifest_only=False):
    """Processes one contiguous, gap-free piece of segments end-to-end:
    its own ChainedVideoReader, motion detector, and EventRecorder, never
    concatenated with any other piece (see _run_stream's docstring for
    why -- splice avoidance and a fresh background model per piece).
    Shared by the static (_run_stream) and dynamic-dispatch
    (_run_stream_dynamic) worker loops, which differ only in *how* they
    decide which piece to hand this function next.

    `frames_done`/`total_events` are the running totals across every piece
    this worker has already finished (for progress reporting); `pbar` is
    the tqdm bar to keep drawing to in the standalone (no progress_queue)
    case, or None otherwise. Returns the updated (frames_done,
    total_events, pbar, manifest_events) -- manifest_events is this
    piece's full list of manifest-only event dicts (empty unless
    manifest_only=True).

    When manifest_only and running under a progress_queue (multi-worker),
    newly-closed events are also pushed to the queue as they're found --
    at each segment boundary, and once more for any trailing remainder
    right before this piece returns -- so the main process can append them
    to the on-disk manifest immediately, the same crash-safety property
    --workers>1 clip mode already gets from per-segment resume state. The
    per-piece return value below is the authoritative complete list (used
    directly by the no-queue --workers 1 path); in the queued path it's
    redundant with what's already on disk by the time this returns.
    """
    gpu_ok = use_gpu and torch_cuda_available()
    use_tqdm = progress_queue is None

    cpu_reader = ChainedVideoReader(piece_paths)
    fps = cpu_reader.fps or 25.0

    detector = (TorchMotionDetector(cpu_reader.width, cpu_reader.height, points)
                if gpu_ok else MotionDetector(cpu_reader.width, cpu_reader.height, points, downscale=downscale))

    reader = cpu_reader
    decode_backend = "CPU (OpenCV)"
    if use_nvdec:
        try:
            reader = FFmpegNVDECReader(piece_paths, cpu_reader.width, cpu_reader.height, fps)
            decode_backend = "GPU (ffmpeg NVDEC)"
        except Exception as e:
            print(f"[motion-scan] {name_prefix}NVDEC decode unavailable ({e}); falling back to CPU decode.")

    pre_frames = max(int(pre * fps), 1)
    post_frames = max(int(post * fps), 1)
    min_frames = max(int(min_event_len * fps), 1)
    recorder = EventRecorder(output_dir, fps, (cpu_reader.width, cpu_reader.height), pre_frames, post_frames,
                             source_name_fn=cpu_reader.filename_for_index,
                             manifest_only=manifest_only, range_fn=cpu_reader.segments_for_range)

    compute_backend = "GPU (PyTorch CUDA)" if gpu_ok else "CPU"
    print(f"[motion-scan] {name_prefix}Compute: {compute_backend} | Decode: {decode_backend} | "
          f"{len(piece_paths)} file(s), ~{cpu_reader.total_frames} frames, {fps:.2f} fps, frame-skip={frame_skip}")

    # Segment boundaries within this piece, so we can tell the main
    # process "segment (path, start, end) is fully read" the moment we
    # cross into the next one -- this is what lets resume work at
    # segment granularity instead of waiting for the whole piece (or
    # chunk) to finish (see "Resume / save-state support" above
    # _state_path()).
    seg_boundaries = []
    cum = 0
    for (seg_path, seg_start, _seg_end, _part_idx, _total_parts), cnt in zip(cpu_reader.segments,
                                                                             cpu_reader.file_frame_counts):
        cum += cnt
        seg_boundaries.append((cum, (seg_path.as_posix(), seg_start, seg_start + cnt)))
    seg_ptr = 0
    manifest_flush_ptr = 0  # index into recorder.manifest_events already pushed to progress_queue

    consecutive_motion = 0
    idx = -1
    last_motion_now = False
    warmup_frames = int(2 * fps)  # background model needs a couple seconds to learn the scene

    # When run standalone (no queue), this process owns the console and
    # can draw its own live-redrawing tqdm bar. When run as one of several
    # parallel workers, it must NOT touch the console directly -- several
    # processes moving the cursor at once just scrambles the screen.
    # Instead it reports progress through `progress_queue`, and the main
    # process (which owns the console) draws a static, per-worker bar
    # from that.
    if use_tqdm and pbar is None:
        pbar = tqdm(
            total=cpu_reader.total_frames or None,
            unit=" frames",
            bar_format="\033[97m{desc} | Progress: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                       "[{elapsed}<{remaining}, {rate_fmt}]\033[0m",
        )
        pbar.set_description_str(f"{name_prefix}Detected: 0")
    report_every = max(int(fps) // 2, 1) if not use_tqdm else None  # ~twice a second of source footage

    while True:
        ok, frame = reader.read()
        if not ok:
            break
        idx += 1
        if frame_skip <= 1 or idx % frame_skip == 0:
            score = detector.score(frame)
            last_motion_now = score >= threshold and idx >= warmup_frames
        motion_now = last_motion_now
        consecutive_motion = consecutive_motion + 1 if motion_now else 0
        confirmed_motion = consecutive_motion >= min_frames
        recorder.push_frame(frame, idx, motion_now, confirmed_motion)
        if use_tqdm:
            pbar.set_description_str(f"{name_prefix}Detected: {total_events + recorder.event_index}")
            pbar.update(1)
        else:
            absolute_idx = frames_done + idx
            if idx % report_every == 0:
                progress_queue.put((name_prefix, absolute_idx + 1, total_events + recorder.event_index,
                                    False, None, []))
            if seg_ptr < len(seg_boundaries) and idx + 1 >= seg_boundaries[seg_ptr][0]:
                new_events = recorder.manifest_events[manifest_flush_ptr:] if manifest_only else []
                manifest_flush_ptr = len(recorder.manifest_events)
                progress_queue.put((name_prefix, absolute_idx + 1, total_events + recorder.event_index,
                                    False, seg_boundaries[seg_ptr][1], new_events))
                seg_ptr += 1

    # finalize() force-closes any clip still open at this piece's end
    # instead of letting it carry into the next piece -- exactly the
    # "cut, don't splice" behavior _run_stream's docstring promises.
    recorder.finalize(idx)
    if not use_tqdm and manifest_only:
        trailing = recorder.manifest_events[manifest_flush_ptr:]
        if trailing:
            progress_queue.put((name_prefix, frames_done + idx + 1, total_events + recorder.event_index,
                                False, None, trailing))
    total_events += recorder.event_index
    frames_done += idx + 1
    reader.release()
    if reader is not cpu_reader:
        cpu_reader.release()
    return frames_done, total_events, pbar, recorder.manifest_events


def _run_stream(pieces, points, output_dir, threshold, min_event_len, pre, post,
                use_gpu, downscale, use_nvdec, frame_skip, name_prefix, progress_queue=None,
                manifest_only=False):
    """Decodes+detects motion across this worker's assigned `pieces` --
    a list of chronologically-ordered file/segment lists, each internally
    gap-free (see the partition step in scan()). A worker normally gets
    exactly one piece (the single-worker path always does: it's called
    with `[paths]`); it only gets more than one when there weren't enough
    workers to give every gap-free stretch its own worker.

    Each piece gets its own ChainedVideoReader, motion detector, and
    EventRecorder, run one after another -- never concatenated into one
    stream. Concatenating them would let ChainedVideoReader silently
    present two temporally-disjoint pieces as continuous frames, and if an
    event's post-roll padding was still open right at that handoff,
    EventRecorder would splice unrelated footage into one clip (the exact
    bug this whole partitioning scheme exists to avoid -- see the
    partition comment in scan()). Starting fresh per piece also resets the
    detector's background model, which matters just as much: judging a
    piece's first few frames against a background model built from a
    different point in time would misfire as motion.
    """
    if progress_queue is not None:
        # We are one of several worker *processes*. OpenCV has its own
        # internal thread pool for decode/per-frame ops; left at its
        # default, each process would compete for threads on top of the
        # parallelism we already get from running N processes, causing CPU
        # oversubscription (every core "busy" but burning cycles on context
        # switches instead of work). Single-worker runs keep OpenCV's
        # default threading since there's no process-level contention then.
        cv2.setNumThreads(1)

    total_events = 0
    frames_done = 0  # cumulative frames across every piece finished so far, for progress reporting
    pbar = None
    manifest_events = []
    for piece_paths in pieces:
        frames_done, total_events, pbar, piece_events = _process_piece(
            piece_paths, points, output_dir, threshold, min_event_len, pre, post,
            use_gpu, downscale, use_nvdec, frame_skip, name_prefix, progress_queue,
            frames_done, total_events, pbar, manifest_only)
        manifest_events.extend(piece_events)

    if progress_queue is None:
        if pbar is not None:
            pbar.close()
    else:
        progress_queue.put((name_prefix, frames_done, total_events, True, None, []))
    return total_events, manifest_events


def _run_stream_dynamic(work_queue, points, output_dir, threshold, min_event_len, pre, post,
                        use_gpu, downscale, use_nvdec, frame_skip, name_prefix, progress_queue,
                        manifest_only=False):
    """Like _run_stream, but instead of a fixed list of pieces decided
    upfront, pulls the next piece from a shared `work_queue` as soon as it
    finishes the one before -- so a worker that turns out faster (or
    happens to draw easier/shorter pieces) keeps picking up more work
    instead of idling once its static share is done while a slower worker
    is still grinding through its own. `work_queue` is pre-loaded with
    every piece before any worker starts (see _build_dispatch_pieces in
    scan()); a worker exits once a non-blocking get finds it empty.

    Each dequeued piece is still processed by _process_piece exactly like
    a static piece -- own reader/detector/recorder, never concatenated --
    so the cut-not-splice guarantee is unaffected; only *which* worker
    gets *which* piece, and *when*, is decided at runtime instead of
    upfront. Always runs with a progress_queue (never the standalone-tqdm
    path), since dynamic dispatch across multiple processes only makes
    sense with `--workers > 1`.
    """
    cv2.setNumThreads(1)  # see _run_stream's comment on the same line

    total_events = 0
    frames_done = 0
    manifest_events = []
    while True:
        try:
            piece_paths = work_queue.get_nowait()
        except queue_mod.Empty:
            break
        frames_done, total_events, _, piece_events = _process_piece(
            piece_paths, points, output_dir, threshold, min_event_len, pre, post,
            use_gpu, downscale, use_nvdec, frame_skip, name_prefix, progress_queue,
            frames_done, total_events, None, manifest_only)
        manifest_events.extend(piece_events)

    progress_queue.put((name_prefix, frames_done, total_events, True, None, []))
    return total_events, manifest_events


def _build_dispatch_pieces(positions, units, target_piece_frames):
    """Cuts the chronological (position, unit) list into contiguous,
    gap-free pieces of at most `target_piece_frames` each, for
    --dynamic's shared work queue (see scan()). Reuses the same hard-gap
    rule the static partition uses (a piece always ends where the next
    unit's true `positions[i]` isn't immediately after the previous one's,
    e.g. at a resume-dropped file) so pieces can never straddle a
    real-time discontinuity -- only *how many* segments a piece holds
    before that (bounded by the target instead of by a worker's share of
    the grand total) differs from the static partition's pieces. Returns
    a flat list of pieces (each a list of segments), in chronological
    order."""
    pieces = []
    cur = []
    cur_frames = 0
    prev_pos = None
    for pos, (length, segment, _orig_idx) in zip(positions, units):
        hard_gap = prev_pos is not None and pos != prev_pos + 1
        if cur and (hard_gap or cur_frames >= target_piece_frames):
            pieces.append(cur)
            cur = []
            cur_frames = 0
        cur.append(segment)
        cur_frames += length
        prev_pos = pos
    if cur:
        pieces.append(cur)
    return pieces


def _scan_dynamic(units, positions, points, output_dir, threshold, min_event_len, pre, post,
                  use_gpu, downscale, use_nvdec, frame_skip, workers, nvdec_workers,
                  state_path, fingerprint, total_segments_all, completed_segments,
                  manifest_only=False, manifest_path=None):
    """--dynamic's worker-management path: instead of pre-assigning each
    worker a fixed, upfront share of the timeline (scan()'s static
    contiguous partition), cut the pending timeline into more, smaller
    gap-free pieces than there are workers (see _build_dispatch_pieces)
    and put them all in one shared queue; each of `workers` processes
    pulls the next piece as soon as it finishes the one before, via
    _run_stream_dynamic. Proposed as a test for whether static
    partitioning was ever leaving throughput on the table (e.g. a worker
    idling once its own share is done while a slower/heavier one is still
    grinding) -- unverified before this flag existed, worth A/B-ing
    against plain --workers rather than assuming either way.

    DISPATCH_PIECES_PER_WORKER pieces per worker on average is a
    trade-off, not a free parameter: more/smaller pieces balance more
    finely but each piece boundary is a place an event can get cut into
    two clips (same trade-off the static partition already accepts at
    worker boundaries -- see scan()'s partition comment), so this
    intentionally stays a small multiple rather than going all the way
    down to one piece per segment.
    """
    import concurrent.futures
    import multiprocessing

    DISPATCH_PIECES_PER_WORKER = 4
    total_frames_pending = sum(u[0] for u in units)
    target_piece_frames = max(total_frames_pending / (workers * DISPATCH_PIECES_PER_WORKER), TARGET_SEGMENT_FRAMES)
    dispatch_pieces = _build_dispatch_pieces(positions, units, target_piece_frames)

    print(f"[motion-scan] Dynamic dispatch: {len(units)} pending segment(s) cut into "
          f"{len(dispatch_pieces)} piece(s), pulled on demand by {workers} worker process(es).")

    manager = multiprocessing.Manager()
    work_queue = manager.Queue()
    for piece in dispatch_pieces:
        work_queue.put(piece)
    progress_queue = manager.Queue()

    idx_width = len(str(workers - 1))
    detected_width = 4

    def _desc(i, events):
        return f"w{i:0{idx_width}d}_Detected:{events:>{detected_width}d}"

    overall_bar = tqdm(
        total=total_frames_pending or None,
        unit=" frames",
        position=0,
        leave=True,
        bar_format="\033[1;96mOVERALL  | Progress: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                   "[{elapsed}<{remaining}, {rate_fmt}]\033[0m",
    )

    # Per-worker bars can't show a percentage/total here -- unlike the
    # static path, a worker's eventual frame share isn't known upfront (it
    # depends on how many pieces it happens to pull before the queue runs
    # dry), so this is a plain count-and-rate readout instead.
    bars = {}
    for i in range(workers):
        prefix = f"w{i}_"
        bar = tqdm(
            total=None,
            unit=" frames",
            position=len(bars) + 1,
            leave=True,
            bar_format="\033[97m{desc} | {n_fmt} frames [{elapsed}, {rate_fmt}]\033[0m",
        )
        bar.set_description_str(_desc(i, 0))
        bars[prefix] = bar

    total_events = 0
    manifest_f = open(manifest_path, "a") if manifest_only else None
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        future_to_idx = {
            pool.submit(_run_stream_dynamic, work_queue, points, output_dir, threshold, min_event_len, pre, post,
                        use_gpu, downscale, use_nvdec or (i < nvdec_workers), frame_skip, f"w{i}_",
                        progress_queue, manifest_only): i
            for i in range(workers)
        }
        done_count = 0
        while done_count < len(future_to_idx):
            try:
                prefix, idx, events, done, completed_segment, new_events = progress_queue.get(timeout=0.5)
            except queue_mod.Empty:
                continue
            bar = bars[prefix]
            bar.set_description_str(_desc(int(prefix[1:-1]), events))
            delta = idx - bar.n
            bar.update(delta)
            overall_bar.update(delta)
            if new_events and manifest_f is not None:
                for ev in new_events:
                    manifest_f.write(json.dumps(ev) + "\n")
                manifest_f.flush()
            if completed_segment is not None:
                completed_segments.add(tuple(completed_segment))
                _save_state(state_path, fingerprint, total_segments=total_segments_all,
                            completed_segments=[list(s) for s in completed_segments])
            if done:
                done_count += 1
        for fut in concurrent.futures.as_completed(future_to_idx):
            total_events += fut.result()[0]

    if manifest_f is not None:
        manifest_f.close()
    for bar in bars.values():
        bar.close()
    overall_bar.close()
    _clear_state(state_path)
    if manifest_only:
        print(f"[motion-scan] Done. {total_events} event(s) logged to {manifest_path} "
              f"({len(units)} segment(s) just processed, {total_segments_all} total including any skipped on resume).")
    else:
        print(f"[motion-scan] Done. {total_events} event clip(s) saved to {output_dir} "
              f"({len(units)} segment(s) just processed, {total_segments_all} total including any skipped on resume).")


def scan(input_path, region_path, output_dir, threshold=0.15, min_event_len=1.0,
         pre=2.0, post=2.0, use_gpu=False, downscale=1, use_nvdec=False, frame_skip=1, workers=1,
         restart=False, nvdec_workers=0, dynamic=False, manifest_only=False):
    paths = discover_videos(input_path)
    with open(region_path) as f:
        region = json.load(f)
    points = region["points"]

    # Header-only probe (no frame decode) to get each file's frame count, so
    # we can balance work by actual size instead of just file count -- and so
    # we know the true number of segments *before* capping --workers below.
    # Capping by len(paths) (the old behavior) badly under-uses --workers
    # once files are split into many fixed-size segments: e.g. 15 files cut
    # into 1000-frame segments can easily yield 2000+ segments, so a request
    # for 19 workers should not get clamped down to 15 just because there
    # happen to be 15 *files*.
    probe = ChainedVideoReader(paths)
    frame_counts = probe.file_frame_counts
    probe.release()

    # A single very large file (e.g. a 1GB chunk of DVR footage) is an
    # indivisible unit under plain file-level balancing -- no matter how the
    # *other* files are arranged, that one file alone can still make its
    # worker the long pole that the whole job waits on. So before balancing,
    # cut every file into fixed-size segments (independent of --workers, so
    # a long file still gets plenty of segments even with few workers,
    # giving the contiguous partition below much finer-grained units to
    # balance with -- which also shortens the tail-off at the end of the
    # run, since a
    # worker that runs out of work just picks up another small segment
    # instead of one whole worker idling while a long-pole straggler
    # finishes). A file shorter than the target is left as a single
    # segment. Trade-off: an event can now get cut at a segment boundary
    # even *within* what used to be one file (on top of the pre-existing
    # cut-at-chunk-boundary trade-off), and a split file's clips get a
    # `_pN` suffix (see ChainedVideoReader.filename_for_index) to avoid
    # name collisions between its segments.
    #
    # Segments are TARGET_SEGMENT_FRAMES each, flat -- a file's remainder
    # (if its frame count isn't an exact multiple) is folded into the last
    # segment instead of being spread evenly across every segment of that
    # file. Evenly redistributing (the old behavior: divide the file into
    # ceil(f/1000) parts, then size each part f/parts) made segment sizes
    # drift away from a round 1000 for every file that didn't divide evenly
    # -- with no benefit, since --workers balancing already operates on
    # actual frame counts per segment, not a fixed size.
    units = []  # each: (length, (path, start, end, part_idx, total_parts), orig_file_idx)
    for i, (path, f) in enumerate(zip(paths, frame_counts)):
        num_parts = max(1, f // TARGET_SEGMENT_FRAMES)
        for part in range(num_parts):
            start = part * TARGET_SEGMENT_FRAMES
            end = f if part == num_parts - 1 else start + TARGET_SEGMENT_FRAMES
            units.append((end - start, (path, start, end, part + 1, num_parts), i))

    total_segments_all = len(units)

    # Now that we know the true segment count, cap --workers by that instead
    # of by file count (see comment above the probe).
    workers = max(1, min(int(workers), total_segments_all))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    state_path = _state_path(output_dir)
    fingerprint = _fingerprint_job(paths, region_path, output_dir, threshold, min_event_len,
                                   pre, post, use_gpu, downscale, use_nvdec, frame_skip, manifest_only)
    saved = None if restart else _load_state(state_path, fingerprint)

    # events_manifest.jsonl is append-only across resumed runs, mirroring
    # the segment-resume state it's written alongside -- a genuine resume
    # (saved is not None) keeps whatever a previous run already flushed and
    # only appends newly-found events; anything else (first run, --restart,
    # or a fingerprint mismatch that already invalidated `saved`) starts
    # from an empty file so it never mixes with a stale/unrelated job.
    manifest_path = Path(output_dir) / "events_manifest.jsonl"
    if manifest_only and saved is None:
        manifest_path.write_text("")

    if workers == 1:
        if saved and 0 in set(saved.get("completed_chunks", [])):
            print(f"[motion-scan] Already completed in a previous run (per {state_path}); nothing to do. "
                  "Pass --restart to force a full re-run.")
            return
        total_events, manifest_events = _run_stream([paths], points, output_dir, threshold, min_event_len, pre, post,
                                   use_gpu, downscale, use_nvdec, frame_skip, name_prefix="",
                                   manifest_only=manifest_only)
        if manifest_only:
            with open(manifest_path, "a") as f:
                for ev in manifest_events:
                    f.write(json.dumps(ev) + "\n")
        _clear_state(state_path)
        if manifest_only:
            print(f"[motion-scan] Done. {total_events} event(s) logged to {manifest_path}")
        else:
            print(f"[motion-scan] Done. {total_events} event clip(s) saved to {output_dir}")
        return

    # Resume support: drop segments already fully completed in a previous
    # run, with a 1-segment-back safety margin per file (see the comment
    # above _state_path()). Completion order within one file is NOT
    # guaranteed to match segment order: a file long enough to span more
    # than one segment can have those segments land in two different
    # (adjacent) worker chunks -- see the partition below -- and those
    # workers can finish at different speeds, so a later segment can be
    # marked done before an earlier one is. An earlier version of this
    # loop treated the first not-done segment as "nothing after this point
    # counts," discarding every already-completed segment past it even
    # when a faster worker had genuinely finished them -- on a job with
    # long, multi-segment files this threw away the bulk of real progress
    # on every resume (confirmed in practice 2026-07-05: a resume credited
    # only ~100/1731 segments despite far more having actually completed).
    # Fixed to only redo segments that are actually not done, plus a
    # 1-segment margin on the done side of every done->pending transition
    # (there can be more than one per file) -- everything else marked done
    # stays skipped regardless of where it sits relative to a gap.
    # `positions` tracks each surviving unit's index in the *original*,
    # unfiltered `units` list -- i.e. its true place in the gapless
    # chronological timeline built above. Resume can drop a whole file (or
    # a file's already-done leading segments) from the *middle* of that
    # timeline while keeping pending segments before and after it, so two
    # units that end up adjacent in `filtered` are not necessarily adjacent
    # in real recording time. The partition step below needs `positions` to
    # tell those two cases apart (see the comment there).
    positions = list(range(len(units)))
    completed_segments = set()
    if saved:
        completed_segments = {tuple(s) for s in saved.get("completed_segments", [])}
    if completed_segments:
        by_file = {}
        for pos, u in zip(positions, units):
            by_file.setdefault(u[2], []).append((pos, u))
        filtered = []
        for orig_idx, file_units in by_file.items():
            keys = [(seg[0].as_posix(), seg[1], seg[2]) for _pos, (_length, seg, _fidx) in file_units]
            done_flags = [k in completed_segments for k in keys]
            redo = [not d for d in done_flags]
            for i in range(len(done_flags) - 1):
                if done_flags[i] and not done_flags[i + 1]:
                    redo[i] = True  # margin: redo the done segment right before each gap
            filtered.extend(fu for fu, r in zip(file_units, redo) if r)
        skipped = total_segments_all - len(filtered)
        positions = [pos for pos, _u in filtered]
        units = [u for _pos, u in filtered]
        if skipped:
            print(f"[motion-scan] Resuming: {skipped}/{total_segments_all} segment(s) already completed "
                  f"in a previous run (per {state_path}), skipping them (redoing one segment of margin per "
                  "file). Pass --restart to force a full re-run.")
        if not units:
            _clear_state(state_path)
            print(f"[motion-scan] Done. All {total_segments_all} segment(s) were already completed; nothing to do.")
            return

    if dynamic:
        _scan_dynamic(units, positions, points, output_dir, threshold, min_event_len, pre, post,
                      use_gpu, downscale, use_nvdec, frame_skip, workers, nvdec_workers,
                      state_path, fingerprint, total_segments_all, completed_segments,
                      manifest_only=manifest_only, manifest_path=manifest_path)
        return

    # Contiguous prefix-sum partition: walk `units` in their existing
    # chronological order (file order, then start offset within a split
    # file -- see the loop that built `units` above) and cut it into
    # `workers` back-to-back runs, each getting roughly total_frames/workers
    # frames. This used to be a greedy longest-processing-time-first (LPT)
    # bin pack (always add the next segment to whichever worker currently
    # has the least total frames), which balances load well but can hand a
    # single worker two segments of the *same* file that aren't adjacent
    # (e.g. segment 42 and segment 59 of a 60-segment file, with segments
    # 43-58 landing on other workers) -- ChainedVideoReader has no way to
    # tell its caller that a big real-time gap sits between two segments it
    # was handed, so if an event's post-roll padding was still open right as
    # the reader crossed that gap, EventRecorder just kept writing whatever
    # frames arrived next, splicing two far-apart moments into one clip
    # (confirmed in practice: a clip's burned-in DVR clock jumped ~12
    # minutes forward mid-clip). A contiguous partition still lets an event
    # get cut into two clips at a genuine chunk boundary -- the pre-existing,
    # accepted trade-off -- but a worker's own segments are now always
    # temporally adjacent, so that boundary is the *only* place a cut (never
    # a splice) can happen.
    #
    # Resume reopens the same risk from a different angle: dropping already-
    # completed segments (see `positions` above) can leave two *pending*
    # units adjacent in `units` even though a whole already-done file (or a
    # done file's leading segments) sits between them in real time. So a
    # worker boundary is forced (`hard_gap`) whenever the next unit's true
    # `positions[i]` isn't immediately after the previous one's -- never let
    # a worker's target frame count alone decide to swallow one of those.
    #
    # If `workers` runs out before every hard gap gets its own worker (more
    # disjoint pending stretches than workers), the leftover gaps land in the
    # same worker as an earlier stretch -- but each worker processes a list
    # of *pieces*, not one flat segment list (`_run_stream` gives every piece
    # its own ChainedVideoReader/EventRecorder, run one after another). So a
    # hard gap always starts a new piece even when it can't get a new worker,
    # meaning two temporally-disjoint stretches assigned to the same worker
    # are queued (processed back-to-back, each finalized before the next
    # starts) rather than concatenated into one stream -- the splice this
    # whole partition exists to prevent stays impossible either way, just at
    # the cost of that worker doing its two pieces sequentially instead of
    # a dedicated process each.
    #
    # `cum` is the running total across *all* units seen so far (never reset
    # per worker) so the target-based advance stays self-correcting: each
    # worker boundary lands at whatever fraction of the grand total `cum`
    # has reached, so a worker that ends up a bit short (e.g. right after a
    # forced hard-gap advance) doesn't throw off every worker after it --
    # the next boundary is still computed from the true running total, not
    # from a fresh per-worker zero.
    total_frames_pending = sum(u[0] for u in units)
    target_per_worker = total_frames_pending / workers if workers else total_frames_pending
    chunk_pieces = [[] for _ in range(workers)]  # chunk_pieces[w]: list of pieces; each piece: list of segments
    chunk_totals = [0] * workers
    w = 0
    cum = 0
    prev_pos = None
    for pos, (length, segment, orig_idx) in zip(positions, units):
        hard_gap = prev_pos is not None and pos != prev_pos + 1
        if w < workers - 1 and hard_gap:
            w += 1
        elif target_per_worker > 0:
            w = max(w, min(int(cum / target_per_worker), workers - 1))
        if hard_gap or not chunk_pieces[w]:
            chunk_pieces[w].append([])
        chunk_pieces[w][-1].append(segment)
        chunk_totals[w] += length
        cum += length
        prev_pos = pos

    chunks = chunk_pieces
    nonempty = [(c, t) for c, t in zip(chunks, chunk_totals) if c]
    chunks = [c for c, _ in nonempty]
    chunk_totals = [t for _, t in nonempty]
    pending = list(range(len(chunks)))

    print(f"[motion-scan] Splitting {len(paths)} file(s) ({len(units)} pending segment(s) of "
          f"{total_segments_all} total) across {len(chunks)} worker process(es) (balanced by frame count).")

    import concurrent.futures
    import multiprocessing

    manager = multiprocessing.Manager()
    progress_queue = manager.Queue()
    # Fixed-width worker index and "Detected" count so every row's {desc}
    # field is exactly the same length -- otherwise w0 vs w18, and a 1-digit
    # vs 3-digit detected count, shift everything after them (the "Progress:
    # NN%|bar|..." part) out of alignment between rows.
    idx_width = len(str(len(chunks) - 1))
    detected_width = 4

    def _desc(i, events):
        return f"w{i:0{idx_width}d}_Detected:{events:>{detected_width}d}"

    # Main/overall bar sits above all the per-worker rows (position 0) and
    # tracks total frames done across every worker combined -- handy for
    # eyeballing true wall-clock progress without having to mentally add up
    # N per-worker bars, and for A/B-ing --workers values (same total work,
    # so the overall bar's ETA/rate is the number to compare across runs).
    overall_total = sum(chunk_totals) or None
    overall_bar = tqdm(
        total=overall_total,
        unit=" frames",
        position=0,
        leave=True,
        bar_format="\033[1;96mOVERALL  | Progress: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                   "[{elapsed}<{remaining}, {rate_fmt}]\033[0m",
    )

    bars = {}
    for i in pending:
        prefix = f"w{i}_"
        bar = tqdm(
            total=chunk_totals[i] or None,
            unit=" frames",
            position=len(bars) + 1,
            leave=True,
            bar_format="\033[97m{desc} | Progress: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                       "[{elapsed}<{remaining}, {rate_fmt}]\033[0m",
        )
        bar.set_description_str(_desc(i, 0))
        bars[prefix] = bar

    total_events = 0
    # Only the main process (here) ever touches the console for progress --
    # workers just push (prefix, frame_idx, events, done, completed_segment)
    # tuples to the queue. That keeps all cursor movement single-owner, so
    # the bars stay static instead of fighting each other across processes.
    # `completed_segment` (a (path, start, end) key, or None) is set the
    # moment a worker finishes reading one of its segments -- we persist
    # that to the state file right away, so resume granularity is "which
    # segments finished," not "which whole chunks finished."
    # `nvdec_workers` lets a caller assign NVDEC decode to only the first N
    # workers (by position in `pending`) instead of --nvdec applying to every
    # worker uniformly -- for testing whether a couple of GPU-decode workers
    # add throughput *alongside* CPU-decode workers rather than replacing
    # them (GPU decode is a separate hardware block from the CPU cores that
    # CPU-decode workers already saturate, so it isn't necessarily subject
    # to the same ceiling -- unverified, this is what the flag is for).
    manifest_f = open(manifest_path, "a") if manifest_only else None
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(pending)) as pool:
        future_to_idx = {
            pool.submit(_run_stream, chunks[i], points, output_dir, threshold, min_event_len, pre, post,
                        use_gpu, downscale, use_nvdec or (pos < nvdec_workers), frame_skip, f"w{i}_",
                        progress_queue, manifest_only): i
            for pos, i in enumerate(pending)
        }
        done_count = 0
        while done_count < len(future_to_idx):
            try:
                prefix, idx, events, done, completed_segment, new_events = progress_queue.get(timeout=0.5)
            except queue_mod.Empty:
                continue
            bar = bars[prefix]
            bar.set_description_str(_desc(int(prefix[1:-1]), events))
            delta = idx - bar.n
            bar.update(delta)
            overall_bar.update(delta)
            if new_events and manifest_f is not None:
                for ev in new_events:
                    manifest_f.write(json.dumps(ev) + "\n")
                manifest_f.flush()
            if completed_segment is not None:
                completed_segments.add(tuple(completed_segment))
                _save_state(state_path, fingerprint, total_segments=total_segments_all,
                            completed_segments=[list(s) for s in completed_segments])
            if done:
                done_count += 1
        for fut in concurrent.futures.as_completed(future_to_idx):
            total_events += fut.result()[0]

    if manifest_f is not None:
        manifest_f.close()
    for bar in bars.values():
        bar.close()
    overall_bar.close()
    _clear_state(state_path)
    if manifest_only:
        print(f"[motion-scan] Done. {total_events} event(s) logged to {manifest_path} "
              f"({len(units)} segment(s) just processed, {total_segments_all} total including any skipped on resume).")
    else:
        print(f"[motion-scan] Done. {total_events} event clip(s) saved to {output_dir} "
              f"({len(units)} segment(s) just processed, {total_segments_all} total including any skipped on resume).")


# --------------------------------------------------------------------------
# Manifest extraction: cut real clips from a --manifest-only run's log,
# using local source footage -- see EventRecorder's manifest_only docstring.
# --------------------------------------------------------------------------

def _extract_one(event, input_dir, output_dir):
    """Cuts a single manifest event into a clip. Module-level (not a closure)
    so it can be pickled and handed to a ProcessPoolExecutor worker."""
    out_path = output_dir / event["clip_name"]
    segments = [(input_dir / name, start, end, 1, 1) for name, start, end in event["segments"]]
    reader = ChainedVideoReader(segments)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, reader.fps or 25.0, (reader.width, reader.height))
    while True:
        ok, frame = reader.read()
        if not ok:
            break
        writer.write(frame)
    writer.release()
    reader.release()


def extract_clips(input_path, manifest_path, output_dir, workers=1):
    """Reads events_manifest.jsonl (one JSON object per line: {"clip_name":
    ..., "segments": [[filename, start_frame, end_frame], ...]}) and cuts
    each event into a real clip in output_dir, reading source files by bare
    filename from input_path.

    Filenames only (not full paths) are stored in the manifest deliberately
    -- it may have been produced on a different machine/OS (a cloud VM, e.g.
    Linux paths under a different mount) than the one running `extract`
    (e.g. Windows, `<folder>/data`), so any absolute or OS-specific path
    baked into the manifest would need translating. `discover_videos` never
    looks past one flat directory level either, so the bare filename is
    already everything needed to resolve it against a *local* --input.

    No motion detection here -- this is pure seek+copy (reusing
    ChainedVideoReader's existing GOP seek-warmup, so no re-run of the
    corrupted-frame risk that motivated it), typically much cheaper per clip
    than the original scan. Unlike scan()'s --workers, events have no
    ordering/continuity constraint between them (each is an independent
    seek+copy of its own file range), so splitting them across a
    ProcessPoolExecutor pool is a plain map with no partitioning logic
    needed. `workers=1` (the default) skips the pool entirely and runs
    in-process, matching the original single-threaded behavior. Existing
    output files are left alone (skipped) unless removed first -- there's
    no --force here yet since nothing has needed it.
    """
    input_dir = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    events = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    if not events:
        print(f"[motion-scan] No events in {manifest_path}; nothing to extract.")
        return

    pending = []
    skipped = 0
    for event in events:
        if (output_dir / event["clip_name"]).exists():
            skipped += 1
        else:
            pending.append(event)
    if skipped:
        print(f"[motion-scan] Skipping {skipped} clip(s) that already exist in {output_dir}.")

    extracted = 0
    if workers <= 1 or len(pending) <= 1:
        for i, event in enumerate(pending, 1):
            _extract_one(event, input_dir, output_dir)
            extracted += 1
            print(f"[motion-scan] [{i}/{len(pending)}] Extracted {event['clip_name']}")
    else:
        import concurrent.futures

        workers = min(workers, len(pending))
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_name = {
                pool.submit(_extract_one, event, input_dir, output_dir): event["clip_name"]
                for event in pending
            }
            for i, fut in enumerate(concurrent.futures.as_completed(future_to_name), 1):
                name = future_to_name[fut]
                fut.result()
                extracted += 1
                print(f"[motion-scan] [{i}/{len(pending)}] Extracted {name}")

    print(f"[motion-scan] Done. {extracted} clip(s) extracted to {output_dir}.")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _resolve_folder_paths(args, need_output):
    """Fills in --input/--region(/--output) from --folder when they weren't
    passed explicitly, using the convention: <folder>/data, <folder>/region.json,
    <folder>/output. Explicit --input/--region/--output always win over the
    --folder default, so existing scripts/commands that pass them directly
    keep working unchanged. Raises a clear error if neither --folder nor the
    individual flag was given."""
    folder = Path(args.folder) if args.folder else None

    input_path = args.input or (str(folder / "data") if folder else None)
    region_path = args.region or (str(folder / "region.json") if folder else None)
    if not input_path:
        raise SystemExit("Need --input, or --folder (which defaults --input to <folder>/data).")
    if not region_path:
        raise SystemExit("Need --region, or --folder (which defaults --region to <folder>/region.json).")

    if not need_output:
        return input_path, region_path, None

    output_path = args.output or (str(folder / "output") if folder else None)
    if not output_path:
        raise SystemExit("Need --output, or --folder (which defaults --output to <folder>/output).")
    return input_path, region_path, output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_region = sub.add_parser("select-region", help="Pick the detection zone on the first frame.")
    p_region.add_argument("--folder", default=None,
                           help="Camera folder containing data/ (source videos) and region.json. "
                                "If given, --input/--region default to <folder>/data and "
                                "<folder>/region.json and don't need to be passed separately.")
    p_region.add_argument("--input", default=None, help="Folder of video files (or a single video file). "
                                                          "Defaults to <folder>/data if --folder is given.")
    p_region.add_argument("--region", default=None, help="Path to save the region JSON file. "
                                                           "Defaults to <folder>/region.json if --folder is given.")

    p_scan = sub.add_parser("scan", help="Scan video files for motion inside a saved region.")
    p_scan.add_argument("--folder", default=None,
                         help="Camera folder containing data/ (source videos), output/ (clips go here), "
                              "and region.json. If given, --input/--region/--output default to "
                              "<folder>/data, <folder>/region.json, and <folder>/output, and don't "
                              "need to be passed separately. Lets you keep one self-contained folder "
                              "per camera/location.")
    p_scan.add_argument("--input", default=None, help="Folder of video files (or a single video file). "
                                                        "Defaults to <folder>/data if --folder is given.")
    p_scan.add_argument("--region", default=None, help="Path to the region JSON file from select-region. "
                                                         "Defaults to <folder>/region.json if --folder is given.")
    p_scan.add_argument("--output", default=None, help="Folder to write event clips into. "
                                                         "Defaults to <folder>/output if --folder is given.")
    p_scan.add_argument("--threshold", type=float, default=0.15,
                        help="Fraction (0-1) of the region that must be foreground to count as motion. Default 0.15.")
    p_scan.add_argument("--min-event-len", type=float, default=1.0,
                        help="Seconds of sustained motion required before starting a new event clip. Default 1.0.")
    p_scan.add_argument("--pre", type=float, default=2.0,
                        help="Seconds of footage to keep before motion starts. Default 2.0.")
    p_scan.add_argument("--post", type=float, default=2.0,
                        help="Seconds of footage to keep after motion stops. Default 2.0.")
    p_scan.add_argument("--downscale", type=int, default=1,
                        help="Downscale factor for the CPU detector only. Default 1 (off).")
    p_scan.add_argument("--gpu", action="store_true",
                        help="Use GPU compute (TorchMotionDetector) instead of the default CPU/MOG2 "
                             "detector. Off by default: on this project's hardware CPU compute measured "
                             "faster at every worker count tested, and the GPU detector's fixed "
                             "diff>4*std threshold misses low-contrast (night/IR) motion that MOG2 "
                             "catches -- see DESIGN.md/PROGRESS.md 2026-07-06 before enabling.")
    p_scan.add_argument("--no-gpu", action="store_true",
                        help="Deprecated: CPU compute is now the default. Kept so existing scripts "
                             "still work; overrides --gpu if both are given.")
    p_scan.add_argument("--nvdec", action="store_true",
                        help="Try ffmpeg NVDEC decode for every worker (off by default -- in testing this "
                             "was SLOWER than plain CPU decode due to GPU->host copy + pipe overhead; only "
                             "enable to re-test on your own footage/ffmpeg build). Independent of --gpu "
                             "(GPU decode and GPU compute are separate knobs) -- use --nvdec-workers instead "
                             "if you only want some workers on NVDEC.")
    p_scan.add_argument("--nvdec-workers", type=int, default=0,
                        help="Use NVDEC decode for only this many of the --workers processes (the rest use "
                             "CPU decode), instead of --nvdec applying to all of them. For testing whether a "
                             "few GPU-decode workers add throughput alongside CPU-decode workers rather than "
                             "competing with them for the same CPU decode capacity. Ignored with --workers 1.")
    p_scan.add_argument("--frame-skip", type=int, default=1,
                        help="Only run the detector every Nth frame; frames in between reuse the last "
                             "motion decision. Default 1 (check every frame).")
    p_scan.add_argument("--workers", type=int, default=None,
                        help="Split the input files across N worker processes to decode/detect them in "
                             f"parallel. Default os.cpu_count()-1 = {max(1, (os.cpu_count() or 2) - 1)} on "
                             "this machine (same convention as censor.py's --workers) -- note this project's "
                             "own benchmarks found CPU throughput flat past ~6-10 workers on the dev "
                             "machine's hardware (see DESIGN.md/PROGRESS.md), so a higher default here isn't "
                             "a claim that more helps everywhere, just a reasonable use-what's-available "
                             "default; pass --workers explicitly to pin a specific value. NOTE: an event "
                             "that straddles a boundary between two workers' file chunks may be split into "
                             "two clips.")
    p_scan.add_argument("--restart", action="store_true",
                        help="Ignore any saved progress from a previous interrupted run of this same job "
                             "(same input/region/output/settings) and start over from scratch. By default, "
                             "a hung/crashed run is auto-resumed: chunks that already finished are skipped.")
    p_scan.add_argument("--dynamic", action="store_true",
                        help="Split pending work into more, smaller pieces than --workers and hand them out "
                             "on demand as each worker finishes its current one, instead of assigning every "
                             "worker a fixed upfront share (the default). Experimental: proposed to test "
                             "whether static partitioning was leaving throughput on the table by letting a "
                             "faster/idle worker sit still while a slower one finishes its own fixed share; "
                             "not yet established either way on this project's footage.")
    p_scan.add_argument("--manifest-only", action="store_true",
                        help="Don't write clip video files -- instead log each detected event's source "
                             "file(s) and frame range to <output>/events_manifest.jsonl (a few KB regardless "
                             "of how much footage was scanned, vs. real clips which can be tens of GB). "
                             "Meant for cloud runs: upload footage to a VM (free), scan it there with this "
                             "flag, download only the tiny manifest back (cloud egress/download is NOT free, "
                             "unlike upload), then run 'extract' locally against the same source footage "
                             "(still on disk -- only a copy was uploaded) to cut the actual clips at zero "
                             "detection cost. See the 'extract' subcommand.")

    p_extract = sub.add_parser("extract", help="Cut real clips from a manifest produced by "
                                                "'scan --manifest-only', reading local source footage.")
    p_extract.add_argument("--folder", default=None,
                            help="Camera folder containing data/ (source videos) and output/ (manifest + "
                                 "where clips get written). If given, --input/--manifest/--output default "
                                 "to <folder>/data, <folder>/output/events_manifest.jsonl, and <folder>/output.")
    p_extract.add_argument("--input", default=None, help="Folder of source video files matching the ones the "
                                                           "manifest-only scan ran against. Defaults to "
                                                           "<folder>/data if --folder is given.")
    p_extract.add_argument("--manifest", default=None, help="Path to events_manifest.jsonl. Defaults to "
                                                              "<folder>/output/events_manifest.jsonl if "
                                                              "--folder is given.")
    p_extract.add_argument("--output", default=None, help="Folder to write extracted clips into. Defaults to "
                                                            "<folder>/output if --folder is given.")
    p_extract.add_argument("--workers", type=int, default=None,
                            help="Number of clips to extract in parallel (process pool). Events are "
                                 f"independent so this is a plain split, no partitioning. Default "
                                 f"os.cpu_count()-1 = {max(1, (os.cpu_count() or 2) - 1)} on this machine.")

    args = parser.parse_args()

    if args.command == "select-region":
        input_path, region_path, _ = _resolve_folder_paths(args, need_output=False)
        select_region(input_path, region_path)
    elif args.command == "scan":
        input_path, region_path, output_path = _resolve_folder_paths(args, need_output=True)
        scan(input_path, region_path, output_path,
             threshold=args.threshold, min_event_len=args.min_event_len,
             pre=args.pre, post=args.post,
             use_gpu=args.gpu and not args.no_gpu, downscale=args.downscale,
             use_nvdec=args.nvdec, frame_skip=args.frame_skip,
             workers=args.workers or max(1, (os.cpu_count() or 2) - 1),
             restart=args.restart, nvdec_workers=args.nvdec_workers, dynamic=args.dynamic,
             manifest_only=args.manifest_only)
    elif args.command == "extract":
        folder = Path(args.folder) if args.folder else None
        input_path = args.input or (str(folder / "data") if folder else None)
        output_path = args.output or (str(folder / "output") if folder else None)
        if not input_path:
            raise SystemExit("Need --input, or --folder (which defaults --input to <folder>/data).")
        if not output_path:
            raise SystemExit("Need --output, or --folder (which defaults --output to <folder>/output).")
        manifest_path = args.manifest or str(Path(output_path) / "events_manifest.jsonl")
        extract_clips(input_path, manifest_path, output_path,
                      workers=args.workers or max(1, (os.cpu_count() or 2) - 1))


if __name__ == "__main__":
    main()
