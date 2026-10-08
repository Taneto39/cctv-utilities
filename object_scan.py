#!/usr/bin/env python3
"""
object_scan.py - Object Scan: search raw recordings directly for a Target
(a detector class, or a class group like `animal` = cat + dog) with a
pretrained YOLO model, looking at keyframes only, and report every
Sighting -- a wall-clock time range in which the Target appears. See
CONTEXT.md for the vocabulary (Target / Hit / Sighting) and
docs/adr/0002-region-does-not-filter-hits.md for why a Region never
discards anything.

Unlike detect_objects.py (Clip Filter: "does this short motion clip contain
a cat at all?"), this works on hours of raw footage with no motion pass
first, and answers "when?".

Setup:
    pip install ultralytics      (ffmpeg/ffprobe must be on PATH)

Usage:
    # Camera folder convention: <folder>/data in, <folder>/region.json (if
    # present) used, <folder>/sightings out:
    python object_scan.py --folder "F:/cam1"

    # Any folder of recordings (searched recursively), e.g. a raw NVR export;
    # results go to <input>/sightings:
    python object_scan.py --input "data/cat"

    # Other Targets (any class the model knows, or a group from CLASS_GROUPS):
    python object_scan.py --input "data/cat" --targets animal,person,car

    # Re-group existing Hits with a different gap -- no re-scan needed,
    # since gap isn't part of what decides a Hit:
    python object_scan.py --input "data/cat" --gap 30

    # Straight from Hikvision/Dahua NVRs, or live (see nvr_scan.py):
    python object_scan.py nvr --camera hik1:3 --from 09:00:00
    python object_scan.py live --camera hik1:3,dahua1:1

Output (<output>/):
    sightings.csv   one row per Sighting, rebuilt after every recording
    images/         one JPEG per Sighting: the best-confidence keyframe, with
                    boxes + raw class/confidence (and the Region outline)
    hits.jsonl      raw Hits, appended as found -- Sightings are derived from
                    this, never recorded directly
    .object_scan_state.json   resume state (which recordings are done)

Limits: the detector needs the subject to be roughly 15-20px+ across in
what it's shown. A kitten ~8px across at 1080p was undetectable at any
setting tried (full frame at 640 / 1280, Region crop, upscaled crop).
"""
import argparse
import csv
import hashlib
import json
import os
import queue
import re
import subprocess
import threading
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from select_region import load_region, natural_key

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".ts", ".dav"}

# A Target name that stands for several detector classes at once. At CCTV
# sizes the detector routinely confuses these (an adult cat ~20px across was
# labelled "dog" in every keyframe it was found in), so asking for just
# "cat" would miss it.
CLASS_GROUPS = {
    "animal": ("cat", "dog"),
}

DEFAULT_GAP_S = 10.0
# Consecutive recordings count as one continuous stretch (so a Sighting can
# span the file boundary) only if the next one starts within this many
# seconds of the previous one ending. Real NVR exports do have holes (a
# 2-minute one in the very first test folder), so contiguity is checked,
# never assumed.
CONTIGUOUS_TOLERANCE_S = 3.0
# With a Region, the detector is shown the Region's bounding box scaled up
# by this factor around its centre -- enough surrounding context for the
# detector to classify the subject, while keeping it far larger (relative
# to the model's input size) than in the full frame.
REGION_CROP_SCALE = 3.0
STATE_NAME = ".object_scan_state.json"


# --------------------------------------------------------------------------
# Recordings and their wall-clock start/end
# --------------------------------------------------------------------------

# rename_cctv_download_v2.py output: 2026-08-24_044612.mp4 (start only)
_RENAMED_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})_(\d{2})(\d{2})(\d{2})")
# Raw NVR export: IP_Camera15_..._20261004144359_20261004145059_319810.mp4
_RAW_NVR_RE = re.compile(r"_(\d{14})_(\d{14})(?:_|$)")


def parse_recording_times(name):
    """(start, end) wall-clock datetimes from a recording's file name;
    either can be None when the name doesn't say."""
    stem = Path(name).stem
    m = _RAW_NVR_RE.search(stem)
    if m:
        return (datetime.strptime(m.group(1), "%Y%m%d%H%M%S"),
                datetime.strptime(m.group(2), "%Y%m%d%H%M%S"))
    m = _RENAMED_RE.search(stem)
    if m:
        return datetime(*map(int, m.groups())), None
    return None, None


def discover_recordings(input_dir, exclude_dir):
    """Every video file under input_dir (recursive), natural-sorted by
    relative path. Files still being downloaded without a video extension
    (e.g. a raw export's in-progress `IP_Camera15_..._1365987`) are skipped
    naturally."""
    root = Path(input_dir)
    exclude = Path(exclude_dir).resolve()
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS and exclude not in p.resolve().parents:
            files.append(p)
    files.sort(key=lambda p: natural_key(p.relative_to(root).as_posix()))
    if not files:
        raise SystemExit(f"No recordings found under {input_dir}")
    return files


def probe_duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return None


def probe_size(path, input_args=()):
    out = subprocess.run(["ffprobe", "-v", "error", *input_args, "-select_streams", "v:0", "-show_entries",
                          "stream=width,height", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True).stdout.strip()
    try:
        w, h = out.split(",")[:2]
        return int(w), int(h)
    except ValueError:
        # no path in the message: for Live Watch it's an RTSP URL with the password in it
        raise RuntimeError("ffprobe found no video stream (unreachable stream, wrong password, "
                           "or not a video file)") from None


# --------------------------------------------------------------------------
# Keyframe sources
# --------------------------------------------------------------------------

_SHOWINFO_RE = re.compile(r"\bn:\s*\d+\s+pts:\s*-?\d+\s+pts_time:(-?[\d.]+)")


def iter_keyframes(path, input_args=()):
    """Yields (offset_seconds, bgr_frame) for every keyframe of a recording.

    Decodes with ffmpeg `-skip_frame nokey` and NVDEC (`-hwaccel cuda`,
    falls back to CPU decode automatically if unavailable). Unlike
    motion_scan.py, where NVDEC benchmarked slower than CPU decode, here it's
    ~3-4x faster: decoding only I-frames removes the frame-threading CPU
    decode relies on (~27 keyframes/s on CPU vs ~80-100/s on NVDEC for 1080p
    HEVC on the dev machine). Offsets come from ffmpeg's `showinfo` filter on
    stderr, so they're in ffmpeg's start-normalised timebase (first frame =
    0) regardless of the container's raw PTS (these PS exports start at
    e.g. 11244s), and line up 1:1 with frames on stdout. `path` can also be
    a stream URL (Live Watch), with e.g. `-rtsp_transport tcp` in
    input_args."""
    w, h = probe_size(path, input_args)
    frame_bytes = w * h * 3
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-v", "info", "-hwaccel", "cuda",
           "-skip_frame", "nokey", *input_args, "-i", str(path), "-an", "-sn", "-vf", "showinfo",
           "-fps_mode", "passthrough", "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pts_q = queue.Queue()

    # stderr drained on its own thread, concurrently with stdout -- same
    # pipe-buffer deadlock censor.py hit reading it only afterwards.
    def _drain():
        for raw in proc.stderr:
            m = _SHOWINFO_RE.search(raw.decode("utf-8", "replace"))
            if m:
                pts_q.put(float(m.group(1)))
        pts_q.put(None)

    t = threading.Thread(target=_drain, daemon=True)
    t.start()
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            offset = pts_q.get()
            if offset is None:
                break
            yield offset, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def grab_keyframe(path, offset):
    """The single keyframe at `offset` (same timebase as iter_keyframes)."""
    cmd = ["ffmpeg", "-v", "error", "-skip_frame", "nokey", "-ss", f"{max(0.0, offset - 0.5):.3f}",
           "-i", str(path), "-frames:v", "1", "-f", "image2pipe", "-vcodec", "bmp", "-"]
    out = subprocess.run(cmd, capture_output=True).stdout
    if not out:
        return None
    return cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)


# --------------------------------------------------------------------------
# Targets, Region crop
# --------------------------------------------------------------------------

def resolve_targets(model_names, spec):
    """{class_id: [target names]} for a comma-separated Target list."""
    name_to_id = {name: idx for idx, name in model_names.items()}
    class_targets = {}
    for target in [t.strip() for t in spec.split(",") if t.strip()]:
        classes = CLASS_GROUPS.get(target, (target,))
        unknown = [c for c in classes if c not in name_to_id]
        if unknown:
            raise SystemExit(f"Target '{target}': class(es) not known to this model: {', '.join(unknown)} "
                             f"(groups: {', '.join(CLASS_GROUPS)})")
        for c in classes:
            class_targets.setdefault(name_to_id[c], []).append(target)
    if not class_targets:
        raise SystemExit("No Targets given.")
    return class_targets


def region_crop_box(shapes, frame_w, frame_h, scale=REGION_CROP_SCALE):
    pts = np.array([p for shape in shapes for p in shape], dtype=float)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hw, hh = (x1 - x0) * scale / 2, (y1 - y0) * scale / 2
    return (int(max(0, cx - hw)), int(max(0, cy - hh)),
            int(min(frame_w, cx + hw)), int(min(frame_h, cy + hh)))


def region_distance(shapes, x, y):
    """0 if (x, y) is inside any Shape, else pixel distance to the nearest
    Shape edge."""
    best = None
    for shape in shapes:
        d = cv2.pointPolygonTest(np.array(shape, dtype=np.float32), (float(x), float(y)), True)
        d = 0.0 if d >= 0 else -d
        best = d if best is None else min(best, d)
    return best


# --------------------------------------------------------------------------
# Resume state
# --------------------------------------------------------------------------

def _file_sig(path):
    st = path.stat()
    return [st.st_size, int(st.st_mtime)]


def fingerprint(class_targets, conf, imgsz, model, region):
    """Everything that decides what counts as a Hit. Gap is deliberately not
    in here -- it only regroups existing Hits into Sightings."""
    payload = {
        "class_targets": {str(k): v for k, v in sorted(class_targets.items())},
        "conf": conf,
        "imgsz": imgsz,
        "model": Path(model).name,
        "region": region["points"] if region else None,
        "crop_scale": REGION_CROP_SCALE if region else None,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_state(path, fp):
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("fingerprint") != fp:
        print("[object-scan] Settings that decide a Hit changed since the last run "
              "(Targets, --conf, --imgsz, --model or Region) -- starting fresh.")
        return None
    return data


def save_state(path, fp, done):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"fingerprint": fp, "done": done}))
    os.replace(tmp, path)


def read_hits(hits_path):
    hits = []
    if hits_path.exists():
        with open(hits_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        hits.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # torn last line from an interrupted write
    return hits


# --------------------------------------------------------------------------
# Scanning
# --------------------------------------------------------------------------

def scan_recording(model, path, rel, class_targets, conf, imgsz, batch_size, crop, hits_file, pbar):
    """Runs the detector over every keyframe of one recording, appending
    Hits to hits_file as each batch completes. Decode runs on a background
    thread so NVDEC + the pipe read overlap with GPU inference."""
    class_ids = sorted(class_targets)
    frames_q = queue.Queue(maxsize=batch_size * 4)
    errors = []

    def _produce():
        try:
            for item in iter_keyframes(path):
                frames_q.put(item)
        except Exception as e:  # surfaced in the main thread below
            errors.append(e)
        finally:
            frames_q.put(None)

    threading.Thread(target=_produce, daemon=True).start()
    n_hits = 0
    n_frames = 0
    last_offset = 0.0
    done = False
    while not done:
        batch = []
        while len(batch) < batch_size:
            item = frames_q.get()
            if item is None:
                done = True
                break
            batch.append(item)
        if not batch:
            break
        offsets = [o for o, _ in batch]
        if crop:
            cx0, cy0, cx1, cy1 = crop
            imgs = [f[cy0:cy1, cx0:cx1] for _, f in batch]
        else:
            cx0 = cy0 = 0
            imgs = [f for _, f in batch]
        results = model.predict(imgs, imgsz=imgsz, conf=conf, classes=class_ids, verbose=False)
        lines = []
        for offset, r in zip(offsets, results):
            for b in r.boxes:
                cls = int(b.cls)
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                box = [round(x1 + cx0, 1), round(y1 + cy0, 1), round(x2 + cx0, 1), round(y2 + cy0, 1)]
                for target in class_targets[cls]:
                    lines.append(json.dumps({"file": rel, "offset": round(offset, 3), "target": target,
                                             "cls": model.names[cls], "conf": round(float(b.conf), 3),
                                             "box": box}))
        if lines:
            hits_file.write("\n".join(lines) + "\n")
            hits_file.flush()
            n_hits += len(lines)
        n_frames += len(batch)
        pbar.update(max(0.0, min(offsets[-1] - last_offset, pbar.total - pbar.n)))
        last_offset = offsets[-1]
    if errors:
        raise errors[0]
    return n_frames, n_hits, last_offset


# --------------------------------------------------------------------------
# Hits -> Sightings
# --------------------------------------------------------------------------

def recording_timeline(root, rels, duration_cache):
    """{rel: (stretch_id, base_datetime_or_None)}. Recordings with a known
    start are chained into one stretch while each starts within
    CONTIGUOUS_TOLERANCE_S of the previous one's end; a recording with an
    unknown start is always its own stretch."""
    known, unknown = [], []
    for rel in rels:
        start, end = parse_recording_times(rel)
        if start is None:
            unknown.append(rel)
            continue
        if end is None:
            if rel not in duration_cache:
                duration_cache[rel] = probe_duration(root / rel)
            d = duration_cache[rel]
            end = start + timedelta(seconds=d) if d else None
        known.append((start, end, rel))
    known.sort()
    out = {}
    stretch = -1
    prev_end = None
    for start, end, rel in known:
        if prev_end is None or (start - prev_end).total_seconds() > CONTIGUOUS_TOLERANCE_S:
            stretch += 1
        out[rel] = (stretch, start)
        prev_end = end or start
    for rel in unknown:
        stretch += 1
        out[rel] = (stretch, None)
    return out


def build_sightings(hits, timeline, gap, region):
    """Merges each Target's Hits into Sightings: consecutive Hits (in time)
    of the same Target, within the same contiguous stretch, at most `gap`
    seconds apart."""
    by_target = {}
    for h in hits:
        if h["file"] not in timeline:
            continue  # recording no longer present
        stretch, base = timeline[h["file"]]
        t = (base - datetime(1970, 1, 1)).total_seconds() + h["offset"] if base else h["offset"]
        by_target.setdefault(h["target"], []).append((stretch, t, h))

    sightings = []
    for target, items in by_target.items():
        items.sort(key=lambda it: (it[0], it[1]))
        cur = None
        for stretch, t, h in items:
            if cur and cur["stretch"] == stretch and t - cur["t_end"] <= gap:
                cur["hits"].append((t, h))
                cur["t_end"] = t
            else:
                cur = {"target": target, "stretch": stretch, "t_start": t, "t_end": t,
                       "hits": [(t, h)], "base": timeline[h["file"]][1]}
                sightings.append(cur)

    for s in sightings:
        hs = [h for _, h in s["hits"]]
        s["first"] = hs[0]
        s["best"] = max(hs, key=lambda h: h["conf"])
        s["n_hits"] = len({(h["file"], h["offset"]) for h in hs})
        s["max_conf"] = s["best"]["conf"]
        s["raw_classes"] = Counter(h["cls"] for h in hs)
        if region:
            s["nearest_px"] = min(region_distance(region["points"], (h["box"][0] + h["box"][2]) / 2,
                                                  h["box"][3]) for h in hs)
        else:
            s["nearest_px"] = None
        if s["base"] is not None:
            epoch = datetime(1970, 1, 1)
            s["start"] = epoch + timedelta(seconds=s["t_start"])
            s["end"] = epoch + timedelta(seconds=s["t_end"])
        else:
            s["start"] = s["end"] = None
    sightings.sort(key=lambda s: (s["start"] or datetime.min, s["first"]["file"], s["first"]["offset"]))
    return sightings


def _image_name(s):
    if s["start"]:
        label = s["start"].strftime("%Y%m%d-%H%M%S")
    else:
        label = f"{Path(s['first']['file']).stem}_{s['first']['offset']:08.1f}s"
    return f"{s['target']}_{label}.jpg"


def render_image(root, s, hits, region, dest):
    best = s["best"]
    frame = grab_keyframe(root / best["file"], best["offset"])
    if frame is None:
        return False
    if region:
        for shape in region["points"]:
            cv2.polylines(frame, [np.array(shape, dtype=np.int32)], True, (0, 255, 255), 1)
    same_frame = [h for h in hits if h["file"] == best["file"] and h["offset"] == best["offset"]
                  and h["target"] == s["target"]]
    for h in same_frame:
        x1, y1, x2, y2 = map(int, h["box"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{h['cls']} {h['conf']:.2f}"
        cv2.putText(frame, label, (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(frame, label, (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    cv2.imwrite(str(dest), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return True


def write_outputs(root, out_dir, rels, hits, gap, region, duration_cache):
    """Rebuilds sightings.csv and images/ from the Hits so far. Images are
    only (re)rendered when missing or when their Sighting's best Hit
    changed; images of Sightings that no longer exist are removed."""
    timeline = recording_timeline(root, rels, duration_cache)
    sightings = build_sightings(hits, timeline, gap, region)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    index_path = img_dir / ".index.json"
    try:
        index = json.loads(index_path.read_text())
    except (OSError, json.JSONDecodeError):
        index = {}
    new_index = {}
    rows = []
    for s in sightings:
        name = _image_name(s)
        key = f"{s['best']['file']}@{s['best']['offset']}"
        dest = img_dir / name
        if index.get(name) != key or not dest.exists():
            if not render_image(root, s, hits, region, dest):
                name = ""
        if name:
            new_index[name] = key
        fmt = "%Y-%m-%d %H:%M:%S"
        rows.append({
            "target": s["target"],
            "start": s["start"].strftime(fmt) if s["start"] else "",
            "end": s["end"].strftime(fmt) if s["end"] else "",
            "duration_s": round(s["t_end"] - s["t_start"], 1),
            "file": s["first"]["file"],
            "offset_s": s["first"]["offset"],
            "hits": s["n_hits"],
            "max_conf": s["max_conf"],
            "raw_classes": ";".join(f"{c}:{n}" for c, n in s["raw_classes"].most_common()),
            "nearest_px": "" if s["nearest_px"] is None else round(s["nearest_px"]),
            "image": name,
        })
    for stale in set(index) - set(new_index):
        try:
            (img_dir / stale).unlink()
        except OSError:
            pass
    index_path.write_text(json.dumps(new_index, indent=1))
    csv_path = out_dir / "sightings.csv"
    tmp = csv_path.with_suffix(".tmp")
    fields = ["target", "start", "end", "duration_s", "file", "offset_s", "hits", "max_conf",
              "raw_classes", "nearest_px", "image"]
    # utf-8-sig so Excel opens it with the right encoding on double-click
    with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, csv_path)
    return sightings


# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", help="Camera folder: recordings in <folder>/data, Region from "
                                         "<folder>/region.json (if present), results in <folder>/sightings.")
    parser.add_argument("--input", help="Folder of recordings (searched recursively). Results default to "
                                        "<input>/sightings, Region to <input>/region.json if present.")
    parser.add_argument("--output", help="Where results go (overrides the default above).")
    parser.add_argument("--region", help="region.json to use (overrides the default above).")
    parser.add_argument("--no-region", action="store_true",
                        help="Ignore any region.json: show the detector the whole frame.")
    parser.add_argument("--targets", default="animal",
                        help=f"Comma-separated Targets: detector classes and/or groups "
                             f"({', '.join(f'{k}={'+'.join(v)}' for k, v in CLASS_GROUPS.items())}). "
                             "Default: animal.")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="Minimum detection confidence for a Hit. Default 0.3 (small CCTV subjects "
                             "measured 0.25-0.67).")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Detector input size. Default 640 with a Region (the crop is already "
                             "small) or 1280 without (a ~20px cat was missed entirely at 640).")
    parser.add_argument("--gap", type=float, default=DEFAULT_GAP_S,
                        help=f"Max seconds between Hits merged into one Sighting. Default {DEFAULT_GAP_S:g}. "
                             "Changing it regroups existing Hits without re-scanning.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model", default="yolo26x.pt",
                        help="Default yolo26x.pt -- found the small cat far more reliably than yolo11x.pt "
                             "(which labelled it \"person\"), see DESIGN.md.")
    parser.add_argument("--restart", action="store_true", help="Ignore saved progress and re-scan everything.")
    args = parser.parse_args()

    if args.folder:
        root = Path(args.input) if args.input else Path(args.folder) / "data"
        out_dir = Path(args.output) if args.output else Path(args.folder) / "sightings"
        default_region = Path(args.folder) / "region.json"
    elif args.input:
        root = Path(args.input)
        out_dir = Path(args.output) if args.output else root / "sightings"
        default_region = root / "region.json"
    else:
        raise SystemExit("Need --folder or --input.")

    region = None
    if not args.no_region:
        region_path = Path(args.region) if args.region else default_region
        if region_path.exists():
            region = load_region(region_path)
        elif args.region:
            raise SystemExit(f"--region {region_path} not found.")
    imgsz = args.imgsz or (640 if region else 1280)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = discover_recordings(root, out_dir)
    rels = [p.relative_to(root).as_posix() for p in paths]

    from ultralytics import YOLO  # deferred: slow import, and --help shouldn't need it
    model = YOLO(args.model)
    class_targets = resolve_targets(model.names, args.targets)
    fp = fingerprint(class_targets, args.conf, imgsz, args.model, region)

    state_path = out_dir / STATE_NAME
    hits_path = out_dir / "hits.jsonl"
    saved = None if args.restart else load_state(state_path, fp)
    done = {}
    if saved:
        sigs = {rel: _file_sig(p) for rel, p in zip(rels, paths)}
        done = {rel: sig for rel, sig in saved["done"].items() if sigs.get(rel) == sig}
    # Keep only Hits from recordings that finished -- a recording interrupted
    # mid-scan gets re-scanned whole, so its partial Hits would duplicate.
    kept = [h for h in read_hits(hits_path) if h["file"] in done] if saved else []
    with open(hits_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(h) + "\n" for h in kept)
    save_state(state_path, fp, done)

    pending = [(rel, p) for rel, p in zip(rels, paths) if rel not in done]
    targets_desc = ", ".join(sorted({t for ts in class_targets.values() for t in ts}))
    crop_desc = "none (whole frame)"
    print(f"[object-scan] {len(paths)} recording(s), {len(done)} already done, {len(pending)} to scan. "
          f"Targets: {targets_desc}. imgsz {imgsz}, conf {args.conf}.")

    duration_cache = {}
    total_s = 0.0
    for rel, p in pending:
        duration_cache[rel] = probe_duration(p)
        total_s += duration_cache[rel] or 0.0

    t0 = time.time()
    total_frames = 0
    with tqdm(total=round(total_s), unit="s", desc="footage", dynamic_ncols=True,
              bar_format="{l_bar}{bar}| {n:.0f}/{total:.0f}s footage [{elapsed}<{remaining}]") as pbar:
        for rel, p in pending:
            crop = None
            if region:
                crop = region_crop_box(region["points"], *probe_size(p))
                crop_desc = f"Region x{REGION_CROP_SCALE:g} -> {crop}"
            with open(hits_path, "a", encoding="utf-8") as hf:
                n_frames, n_hits, last = scan_recording(model, p, rel, class_targets, args.conf, imgsz,
                                                        args.batch_size, crop, hf, pbar)
            # top up to the probed duration (last keyframe isn't the very end);
            # clamped since keyframe offsets can overshoot the probed duration slightly
            pbar.update(max(0.0, min((duration_cache[rel] or last) - last, pbar.total - pbar.n)))
            total_frames += n_frames
            done[rel] = _file_sig(p)
            save_state(state_path, fp, done)
            tqdm.write(f"  {rel}: {n_frames} keyframes, {n_hits} hit(s)")
            write_outputs(root, out_dir, rels, read_hits(hits_path), args.gap, region, duration_cache)

    elapsed = time.time() - t0
    sightings = write_outputs(root, out_dir, rels, read_hits(hits_path), args.gap, region, duration_cache)
    if total_frames:
        print(f"[object-scan] Scanned {total_frames} keyframes in {elapsed:.0f}s "
              f"({total_frames / elapsed:.1f} keyframes/s). Detector area: {crop_desc}.")
    per_target = Counter(s["target"] for s in sightings)
    print(f"[object-scan] {len(sightings)} Sighting(s) "
          f"({', '.join(f'{t}: {n}' for t, n in per_target.items()) or 'none'}) -> {out_dir / 'sightings.csv'}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in ("nvr", "live", "select-region"):
        import nvr_scan  # NVR source + Live Watch (needs nvr-sdk, see .env.example)
        nvr_scan.main(sys.argv[1:])
    else:
        main()
