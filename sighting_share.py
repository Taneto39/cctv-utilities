#!/usr/bin/env python3
"""
sighting_share.py - copies each Sighting's picture into a destination
folder as it is found, e.g. a Google Drive / OneDrive / LINE-synced folder
to share live with someone else. Copies the clean keyframe from frames/
(no boxes or region outline), not the annotated picture in images/ -- the
images/.index.json written by the scan says which frame each picture was
drawn from. Same input as the Sighting Wall (every camera folder under one
root); like the Wall it only reads, so stopping it never affects a scan.
No window -- runs headless.

    python sighting_share.py --dest /path/to/shared-folder
    python sighting_share.py --dest /path/to/shared-folder --live-only --camera hik1:3

Only Sightings that end after this script starts are copied unless --all
is given. Files are named <camera>_<image name>; a Sighting that later gets
a better picture overwrites its own copy. Copies are written to a temp
name and renamed, so a sync client never uploads a half-written file.
Ctrl+C to stop.
"""
import argparse
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import nvr_scan
from sighting_wall import _read_csv


def _sightings(root, cameras):
    for csv_path in sorted(Path(root).glob("*/sightings/sightings.csv")):
        camera = csv_path.parent.parent.name
        if cameras and camera not in cameras:
            continue
        try:
            index = json.loads((csv_path.parent / "images" / ".index.json").read_text())
        except (OSError, json.JSONDecodeError):
            index = {}
        for r in _read_csv(csv_path):
            if not r.get("image"):
                continue
            try:
                r["end"] = datetime.strptime(r["end"], "%Y-%m-%d %H:%M:%S")
            except (KeyError, ValueError):
                continue
            frame = index.get(r["image"])
            if not frame:    # index not written yet: picked up next poll
                continue
            r["camera"] = camera
            r["frame"] = frame
            r["frame_path"] = csv_path.parent / "frames" / frame
            yield r


def _same_file(src, dest):
    """True when dest already holds src -- size match is enough to tell a
    clean frame from an older annotated copy of the same Sighting."""
    try:
        return src.stat().st_size == dest.stat().st_size
    except OSError:
        return False


def _copy(src, dest):
    tmp = dest.with_name(dest.name + ".part")
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dest)
        return True
    except OSError:          # source missing or locked: retried next poll
        try:
            tmp.unlink()
        except OSError:
            pass
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", required=True, help="Folder to copy Sighting pictures into (created if missing).")
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

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    # Accept object_scan's "hik1:3" form as well as the folder name "hik1_ch3".
    cameras = {c.strip().replace(":", "_ch") for c in args.camera.split(",") if c.strip()}
    for c in sorted(cameras):
        if not (Path(args.root) / c / "sightings").is_dir():
            print(f"[sighting-share] warning: no Sightings folder for camera {c!r} under {args.root} (yet)",
                  flush=True)
    since = None if args.all else datetime.now().replace(microsecond=0)
    copied = {}              # dest path -> frame name it was copied from
    print(f"[sighting-share] {Path(args.root).resolve()} -> {dest.resolve()} -- Ctrl+C to stop", flush=True)
    try:
        while True:
            for r in _sightings(args.root, cameras):
                if since is not None and r["end"] < since:
                    continue
                if args.live_only and "live" not in r.get("source", ""):
                    continue
                out = dest / f"{r['camera']}_{r['image']}"
                if copied.get(out) == r["frame"]:
                    continue
                if out not in copied and _same_file(r["frame_path"], out):
                    copied[out] = r["frame"]     # already copied by an earlier run
                    continue
                if _copy(r["frame_path"], out):
                    print(f"[sighting-share] {'updated' if out in copied else 'new'}: {out.name}", flush=True)
                    copied[out] = r["frame"]
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
