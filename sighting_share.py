#!/usr/bin/env python3
"""
sighting_share.py - copies each Sighting's picture into a destination
folder as it is found, e.g. a Google Drive / OneDrive / LINE-synced folder
to share live with someone else. Two versions of every picture:

    <dest>/raw/       the clean keyframe from frames/ (nothing drawn on it)
    <dest>/overlay/   the same keyframe with the Region outline and the
                      Hit boxes, but no class/confidence text

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
from object_scan import read_hits
from select_region import load_region
from sighting_wall import _read_csv

FMT = "%Y-%m-%d %H:%M:%S"


def _sightings(root, cameras):
    for csv_path in sorted(Path(root).glob("*/sightings/sightings.csv")):
        camera = csv_path.parent.parent.name
        if cameras and camera not in cameras:
            continue
        try:
            index = json.loads((csv_path.parent / "images" / ".index.json").read_text())
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
        self.cache = {}      # camera -> (hits files' mtimes, region, hits by frame)

    def _load(self, camera):
        cam_dir = self.root / camera
        paths = [cam_dir / "sightings" / f"hits_{src}.jsonl" for src in ("nvr", "live")]
        key = tuple(p.stat().st_mtime if p.exists() else 0 for p in paths)
        cached = self.cache.get(camera)
        if cached and cached[0] == key:
            return cached[1], cached[2]
        region_path = cam_dir / "region.json"
        region = load_region(region_path) if region_path.exists() else None
        by_frame = {}
        for p in paths:
            for h in read_hits(p):
                by_frame.setdefault(h.get("frame"), []).append(h)
        self.cache[camera] = (key, region, by_frame)
        return region, by_frame

    def render(self, r):
        frame = cv2.imread(str(r["frame_path"]))
        if frame is None:
            return None
        region, by_frame = self._load(r["camera"])
        if region:
            for shape in region["points"]:
                cv2.polylines(frame, [np.array(shape, dtype=np.int32)], True, (0, 255, 255), 1)
        for h in by_frame.get(r["frame"], []):
            if h.get("target") == r["target"]:
                x1, y1, x2, y2 = map(int, h["box"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return frame


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
                    help="Folder to copy Sighting pictures into; raw/ and overlay/ are created inside it.")
    ap.add_argument("--root", default=str(nvr_scan.default_root()),
                    help="Folder of camera folders (default: NVR_SCAN_ROOT in .env, else ./nvr_scans).")
    ap.add_argument("--camera", default="",
                    help="Comma-separated cameras to copy, as hik1:3 or hik1_ch3; default all.")
    ap.add_argument("--live-only", action="store_true",
                    help="Only copy Sightings seen by Live Watch (source live or nvr+live).")
    ap.add_argument("--all", action="store_true",
                    help="Also copy Sightings that ended before this script started.")
    ap.add_argument("--interval", type=float, default=2.0, help="Seconds between checks for new Sightings.")
    args = ap.parse_args()

    raw_dir, overlay_dir = Path(args.dest) / "raw", Path(args.dest) / "overlay"
    for d in (raw_dir, overlay_dir):
        d.mkdir(parents=True, exist_ok=True)
    # Accept object_scan's "hik1:3" form as well as the folder name "hik1_ch3".
    cameras = {c.strip().replace(":", "_ch") for c in args.camera.split(",") if c.strip()}
    for c in sorted(cameras):
        if not (Path(args.root) / c / "sightings").is_dir():
            print(f"[sighting-share] warning: no Sightings folder for camera {c!r} under {args.root} (yet)",
                  flush=True)
    since = None if args.all else datetime.now().replace(microsecond=0)
    overlay = Overlay(args.root)
    done = {}                # file name -> frame name both versions were made from
    print(f"[sighting-share] {Path(args.root).resolve()} -> {Path(args.dest).resolve()} "
          f"(raw/ + overlay/) -- Ctrl+C to stop", flush=True)
    try:
        while True:
            for r in _sightings(args.root, cameras):
                if since is not None and r["end"] < since:
                    continue
                if args.live_only and "live" not in r.get("source", ""):
                    continue
                name = r["name"]
                if done.get(name) == r["frame"]:
                    continue
                raw_out, overlay_out = raw_dir / name, overlay_dir / name
                if name not in done and _same_file(r["frame_path"], raw_out) and overlay_out.exists():
                    done[name] = r["frame"]      # already copied by an earlier run
                    continue
                img = overlay.render(r)
                if img is None or not _write(img, overlay_out) or not _copy(r["frame_path"], raw_out):
                    continue
                print(f"[sighting-share] {'updated' if name in done else 'new'}: {name}", flush=True)
                done[name] = r["frame"]
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
