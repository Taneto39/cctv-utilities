#!/usr/bin/env python3
"""
sighting_share.py - copies each Sighting's picture into a destination
folder as it is found, e.g. a Google Drive / OneDrive / LINE-synced folder
to share live with someone else. Three versions of every picture:

    <dest>/raw/       the clean keyframe from frames/ (nothing drawn on it)
    <dest>/overlay/   the same keyframe with the Region outline and the
                      Hit boxes, but no class/confidence text
    <dest>/crop/      the clean keyframe cut down to the Hit boxes plus a
                      margin (--crop-pad, at least --crop-min px a side),
                      so a small, distant animal is visible on a phone;
                      Hits far enough apart that their crops don't overlap
                      get one file each (<name>-1.jpg, -2, ... left to right)

The images/.index.json written by the scan says which frame each Sighting's
picture was drawn from. Same input as the Sighting Wall (every camera
folder under one root); like the Wall it only reads, so stopping it never
affects a scan. No window -- runs headless.

    python sighting_share.py --dest /path/to/shared-folder
    python sighting_share.py --dest /path/to/shared-folder --live-only --camera hik1:3

Only Sightings that end after this script starts are copied unless --all
is given. Files are named <start date>-<start time>-<camera>.jpg (e.g.
20261007-193532-hik1_ch8.jpg; -<target> is appended only when two Targets
start in the same second on one camera); a Sighting that later gets a
better picture overwrites its own files. Files are written to a temp name
and renamed, so a sync client never uploads a half-written file.
Ctrl+C to stop.
"""
import argparse
import json
import os
import shutil
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import nvr_scan
from select_region import load_region
from sighting_wall import _read_csv

FMT = "%Y-%m-%d %H:%M:%S"


def _stamp(path):
    try:
        st = path.stat()
        return st.st_mtime_ns, st.st_size
    except OSError:
        return None


def _sightings(root, cameras, cache):
    """Every camera's Sighting rows. `cache` (camera -> (stamps, rows)) skips
    re-parsing a camera whose sightings.csv and index haven't changed, so a
    poll stays cheap however many Sightings have piled up."""
    for csv_path in sorted(Path(root).glob("*/sightings/sightings.csv")):
        camera = csv_path.parent.parent.name
        if cameras and camera not in cameras:
            continue
        index_path = csv_path.parent / "images" / ".index.json"
        stamps = (_stamp(csv_path), _stamp(index_path))
        cached = cache.get(camera)
        if cached and cached[0] == stamps:
            yield from cached[1]
            continue
        rows = list(_parse_sightings(csv_path, index_path, camera))
        cache[camera] = (stamps, rows)
        yield from rows


def _parse_sightings(csv_path, index_path, camera):
    try:
        index = json.loads(index_path.read_text())
    except (OSError, json.JSONDecodeError):
        index = {}
    rows = []
    for r in _read_csv(csv_path):
        frame = index.get(r.get("image", ""))
        if not frame:    # no picture, or index not written yet: picked up next poll
            continue
        try:
            r["start"] = datetime.strptime(r["start"], FMT)
            r["end"] = datetime.strptime(r["end"], FMT)
        except (KeyError, ValueError):
            continue
        r["camera"] = camera
        r["frame"] = frame
        r["frame_path"] = csv_path.parent / "frames" / frame
        rows.append(r)
    clash = Counter(r["start"] for r in rows)
    for r in rows:
        name = f"{r['start']:%Y%m%d-%H%M%S}-{camera}"
        r["name"] = (name if clash[r["start"]] == 1 else f"{name}-{r['target']}") + ".jpg"
        yield r


class Overlay:
    """Draws a Sighting's Region outline + Hit boxes (no labels), the same
    marks the scan puts on images/ minus the text."""

    def __init__(self, root):
        self.root = Path(root)
        self.cache = {}      # camera -> {region stamp, region, read position per hits file, hits by frame}

    def _load(self, camera):
        """Hits files are append-only while a scan runs, so only the bytes
        added since the last call are parsed -- re-reading the whole file on
        every append (Live Watch appends constantly) got slower the longer
        this ran. Starts over when a file shrinks or is replaced (the scan
        moves hits_nvr.jsonl aside to a .bak when its settings change)."""
        cam_dir = self.root / camera
        c = self.cache.setdefault(camera, {"region_stamp": None, "region": None, "pos": {}, "by_frame": {}})
        region_path = cam_dir / "region.json"
        stamp = _stamp(region_path)
        if stamp != c["region_stamp"]:
            c["region"] = load_region(region_path) if stamp else None
            c["region_stamp"] = stamp
        files = {}
        for src in ("nvr", "live"):
            p = cam_dir / "sightings" / f"hits_{src}.jsonl"
            try:
                st = p.stat()
                files[p] = (st.st_ino, st.st_size)
            except OSError:
                files[p] = (None, 0)
        if any(p in c["pos"] and (ino != c["pos"][p][0] or size < c["pos"][p][1])
               for p, (ino, size) in files.items()):
            c["pos"], c["by_frame"] = {}, {}
        for p, (ino, size) in files.items():
            pos = c["pos"].get(p, (ino, 0))[1]
            if size <= pos:
                continue
            try:
                with open(p, "rb") as f:
                    f.seek(pos)
                    data = f.read(size - pos)
            except OSError:
                continue
            end = data.rfind(b"\n") + 1     # a torn last line waits for the next call
            for line in data[:end].splitlines():
                try:
                    h = json.loads(line)
                except ValueError:
                    continue
                c["by_frame"].setdefault(h.get("frame"), []).append(h)
            c["pos"][p] = (ino, pos + end)
        return c["region"], c["by_frame"]

    def boxes(self, r):
        _, by_frame = self._load(r["camera"])
        return [list(map(int, h["box"])) for h in by_frame.get(r["frame"], [])
                if h.get("target") == r["target"]]

    def render(self, r, frame):
        frame = frame.copy()
        region, _ = self._load(r["camera"])
        if region:
            for shape in region["points"]:
                cv2.polylines(frame, [np.array(shape, dtype=np.int32)], True, (0, 255, 255), 1)
        for x1, y1, x2, y2 in self.boxes(r):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return frame


def _window(box, pad, min_side, fw, fh):
    """`box` grown by `pad` x its longer side on every side, widened to at
    least `min_side` px each way, and slid (not shrunk) to stay inside the
    frame."""
    x1, y1, x2, y2 = box
    m = pad * max(x2 - x1, y2 - y1)
    out = []
    for lo, hi, size in ((x1 - m, x2 + m, fw), (y1 - m, y2 + m, fh)):
        side = min(size, max(hi - lo, min_side))
        lo = min(max(0, (lo + hi - side) / 2), size - side)
        out.append((int(lo), int(lo + side)))
    (cx1, cx2), (cy1, cy2) = out
    return cx1, cy1, cx2, cy2


def _overlap(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def crops(frame, boxes, pad, min_side):
    """Cut-outs of `frame` around `boxes`, left to right. Boxes whose crop
    windows overlap are merged (window of their union) until none overlap,
    so animals close together share one picture and far-apart ones get
    their own. The whole frame when there are no boxes."""
    if not boxes:
        return [frame]
    fh, fw = frame.shape[:2]
    groups = [list(b) for b in boxes]
    merged = True
    while merged:
        merged = False
        wins = [_window(g, pad, min_side, fw, fh) for g in groups]
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if _overlap(wins[i], wins[j]):
                    a, b = groups[i], groups.pop(j)
                    groups[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    merged = True
                    break
            if merged:
                break
    wins = sorted(_window(g, pad, min_side, fw, fh) for g in groups)
    return [frame[y1:y2, x1:x2] for x1, y1, x2, y2 in wins]


def _crop_names(name, n):
    stem = name[:-len(".jpg")]
    return [name] if n == 1 else [f"{stem}-{i}.jpg" for i in range(1, n + 1)]


def _old_crops(crop_dir, name):
    """crop/ files already written for this Sighting: <name> or <stem>-<n>.jpg."""
    stem = name[:-len(".jpg")]
    return [p for p in crop_dir.glob(f"{stem}*.jpg")
            if p.name == name or p.name[len(stem) + 1:-len(".jpg")].isdigit()]


def _same_file(src, dest):
    """True when dest already holds src -- a size match is enough to tell
    the right frame from an older picture of the same Sighting."""
    try:
        return src.stat().st_size == dest.stat().st_size
    except OSError:
        return False


def _discard(tmp):
    try:
        tmp.unlink()
    except OSError:
        pass


def _copy(src, dest):
    tmp = dest.with_name(dest.name + ".part")
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dest)
        return True
    except OSError:          # source missing or locked: retried next poll
        _discard(tmp)
        return False


def _write(img, dest):
    tmp = dest.with_name(dest.name + ".part.jpg")
    try:
        if not cv2.imwrite(str(tmp), img, [cv2.IMWRITE_JPEG_QUALITY, 90]):
            return False
        os.replace(tmp, dest)
        return True
    except OSError:
        _discard(tmp)
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", required=True,
                    help="Folder to copy Sighting pictures into; raw/, overlay/ and crop/ are created inside it.")
    ap.add_argument("--root", default=str(nvr_scan.default_root()),
                    help="Folder of camera folders (default: NVR_SCAN_ROOT in .env, else ./nvr_scans).")
    ap.add_argument("--camera", default="",
                    help="Comma-separated cameras to copy, as hik1:3 or hik1_ch3; default all.")
    ap.add_argument("--live-only", action="store_true",
                    help="Only copy Sightings seen by Live Watch (source live or nvr+live).")
    ap.add_argument("--all", action="store_true",
                    help="Also copy Sightings that ended before this script started.")
    ap.add_argument("--crop-pad", type=float, default=0.5,
                    help="crop/ margin on every side, as a multiple of the boxes' longer side (default 0.5).")
    ap.add_argument("--crop-min", type=int, default=150,
                    help="crop/ is at least this many px wide and tall (default 150), for context around tiny Hits.")
    ap.add_argument("--interval", type=float, default=2.0, help="Seconds between checks for new Sightings.")
    args = ap.parse_args()

    dest = Path(args.dest)
    raw_dir, overlay_dir, crop_dir = dest / "raw", dest / "overlay", dest / "crop"
    for d in (raw_dir, overlay_dir, crop_dir):
        d.mkdir(parents=True, exist_ok=True)
    # Accept object_scan's "hik1:3" form as well as the folder name "hik1_ch3".
    cameras = {c.strip().replace(":", "_ch") for c in args.camera.split(",") if c.strip()}
    for c in sorted(cameras):
        if not (Path(args.root) / c / "sightings").is_dir():
            print(f"[sighting-share] warning: no Sightings folder for camera {c!r} under {args.root} (yet)",
                  flush=True)
    since = None if args.all else datetime.now().replace(microsecond=0)
    overlay = Overlay(args.root)
    sightings_cache = {}
    missing = set()          # frame paths found gone; a frame is written before the index names it
    done = {}                # file name -> frame name every version was made from
    print(f"[sighting-share] {Path(args.root).resolve()} -> {Path(args.dest).resolve()} "
          f"(raw/ + overlay/ + crop/) -- Ctrl+C to stop", flush=True)
    try:
        while True:
            for r in _sightings(args.root, cameras, sightings_cache):
                if since is not None and r["end"] < since:
                    continue
                if args.live_only and "live" not in r.get("source", ""):
                    continue
                name = r["name"]
                if done.get(name) == r["frame"]:
                    continue
                raw_out, overlay_out = raw_dir / name, overlay_dir / name
                if (name not in done and _same_file(r["frame_path"], raw_out)
                        and overlay_out.exists() and _old_crops(crop_dir, name)):
                    done[name] = r["frame"]      # already copied by an earlier run
                    continue
                if r["frame_path"] in missing:
                    continue
                if not r["frame_path"].exists():
                    # Frame deleted or moved out (e.g. old frames cleared to save
                    # space): the index still names it until the next scan
                    # rebuilds it, so skip quietly instead of retrying every poll.
                    missing.add(r["frame_path"])
                    continue
                frame = cv2.imread(str(r["frame_path"]))
                if frame is None:
                    continue
                cuts = crops(frame, overlay.boxes(r), args.crop_pad, args.crop_min)
                crop_outs = [crop_dir / n for n in _crop_names(name, len(cuts))]
                stale = [p for p in _old_crops(crop_dir, name) if p not in crop_outs]
                if (not _write(overlay.render(r, frame), overlay_out)
                        or not all(_write(c, p) for c, p in zip(cuts, crop_outs))
                        or not _copy(r["frame_path"], raw_out)):
                    continue
                for p in stale:      # the new picture split into fewer crops
                    _discard(p)
                print(f"[sighting-share] {'updated' if name in done else 'new'}: {name}", flush=True)
                done[name] = r["frame"]
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
