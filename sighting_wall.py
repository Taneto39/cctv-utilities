#!/usr/bin/env python3
"""
sighting_wall.py - Sighting Wall: a 2x2 on-screen grid of Sightings as they
are found, from every Object Scan (`object_scan.py nvr`) and Live Watch
(`object_scan.py live`) writing under the same root -- any camera, recorded
or live. Purely a viewer: it only reads each camera's sightings.csv +
images/, so closing it never affects a scan (see CONTEXT.md).

    python sighting_wall.py                   # root from NVR_SCAN_ROOT in .env, else ./nvr_scans
    python sighting_wall.py --root F:/nvr-scans

Each new Sighting takes the next tile in turn (1 -> 2 -> 3 -> 4 -> 1); a
Sighting that grows or gets a better picture updates its own tile instead.
The tile changed most recently has a coloured border. On start the wall
shows the 4 latest existing Sightings. Keys: q / Esc = quit, f = fullscreen.
"""
import argparse
import csv
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

import nvr_scan

N_TILES = 4
HEADER_H = 34
COLORS = {"live": (60, 60, 230), "nvr": (200, 160, 60), "nvr+live": (60, 180, 230)}


def _read_csv(path):
    for _ in range(5):
        try:
            with open(path, newline="", encoding="utf-8-sig") as f:
                return list(csv.DictReader(f))
        except PermissionError:   # a scan is replacing it right now
            time.sleep(0.05)
        except OSError:
            return []
    return []


class Wall:
    def __init__(self, root, gap):
        self.root = Path(root)
        self.gap = timedelta(seconds=gap)
        self.tiles = [None] * N_TILES
        self.next = 0
        self.last_changed = None
        self.seen = []          # every Sighting already shown or skipped: dicts with cam/target/start/end
        self.mtimes = {}
        self.first_poll = True

    def _same(self, a, b):
        return (a["camera"] == b["camera"] and a["target"] == b["target"]
                and a["start"] <= b["end"] + self.gap and b["start"] <= a["end"] + self.gap)

    def _rows(self):
        out = []
        changed = False
        for csv_path in sorted(self.root.glob("*/sightings/sightings.csv")):
            try:
                m = csv_path.stat().st_mtime
            except OSError:
                continue
            if self.mtimes.get(csv_path) != m:
                self.mtimes[csv_path] = m
                changed = True
            for r in _read_csv(csv_path):
                try:
                    r["start"] = datetime.strptime(r["start"], "%Y-%m-%d %H:%M:%S")
                    r["end"] = datetime.strptime(r["end"], "%Y-%m-%d %H:%M:%S")
                except (KeyError, ValueError):
                    continue
                r["image_path"] = csv_path.parent / "images" / r["image"] if r.get("image") else None
                out.append(r)
        return out, changed

    def poll(self):
        rows, changed = self._rows()
        if not changed and not self.first_poll:
            return False
        if self.first_poll:
            self.first_poll = False
            self.seen = list(rows)
            latest = sorted(rows, key=lambda r: r["end"])[-N_TILES:]
            for r in latest:
                self._place(r)
            return True
        updated = False
        new = []
        for r in rows:
            for i, t in enumerate(self.tiles):
                if t and self._same(t, r):
                    if (r["end"], r["start"], r.get("image")) != (t["end"], t["start"], t.get("image")):
                        self.tiles[i] = dict(r, img=None)
                        self.last_changed = i
                        updated = True
                    break
            else:
                match = next((s for s in self.seen if self._same(s, r)), None)
                if match is not None:
                    match.update(start=r["start"], end=r["end"])
                else:
                    new.append(r)
        for r in sorted(new, key=lambda r: r["start"]):
            self.seen.append(r)
            self._place(r)
            updated = True
        return updated

    def _place(self, r):
        self.tiles[self.next] = dict(r, img=None)
        self.last_changed = self.next
        self.next = (self.next + 1) % N_TILES

    def render(self, w, h):
        canvas = np.full((h, w, 3), 24, np.uint8)
        tw, th = w // 2, h // 2
        for i, t in enumerate(self.tiles):
            x0, y0 = (i % 2) * tw, (i // 2) * th
            cv2.putText(canvas, str(i + 1), (x0 + 8, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 90, 90), 2)
            if t is None:
                continue
            if t["img"] is None and t["image_path"] is not None:
                t["img"] = cv2.imread(str(t["image_path"]))   # None while being rewritten: retried next frame
            img = t["img"]
            area_h = th - HEADER_H
            if img is not None:
                s = min(tw / img.shape[1], area_h / img.shape[0])
                iw, ih = int(img.shape[1] * s), int(img.shape[0] * s)
                ox, oy = x0 + (tw - iw) // 2, y0 + HEADER_H + (area_h - ih) // 2
                canvas[oy:oy + ih, ox:ox + iw] = cv2.resize(img, (iw, ih), interpolation=cv2.INTER_AREA)
            src = t.get("source", "nvr")
            color = COLORS.get(src, (200, 200, 200))
            cv2.rectangle(canvas, (x0, y0), (x0 + tw - 1, y0 + HEADER_H), color, -1)
            tag = "LIVE" if src == "live" else "REC" if src == "nvr" else "REC+LIVE"
            label = (f"{i + 1}  {t['camera']}  {t['target']}  "
                     f"{t['start']:%m-%d %H:%M:%S}-{t['end']:%H:%M:%S}  [{tag}]")
            cv2.putText(canvas, label, (x0 + 8, y0 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(canvas, label, (x0 + 8, y0 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            if i == self.last_changed:
                cv2.rectangle(canvas, (x0 + 1, y0 + 1), (x0 + tw - 2, y0 + th - 2), (0, 220, 255), 3)
        cv2.line(canvas, (tw, 0), (tw, h), (60, 60, 60), 2)
        cv2.line(canvas, (0, th), (w, th), (60, 60, 60), 2)
        return canvas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(nvr_scan.default_root()),
                    help="Folder of camera folders (default: NVR_SCAN_ROOT in .env, else ./nvr_scans).")
    ap.add_argument("--interval", type=float, default=1.0, help="Seconds between checks for new Sightings.")
    ap.add_argument("--gap", type=float, default=nvr_scan.osc.DEFAULT_GAP_S,
                    help="Sightings closer than this are treated as the same one (match the scans' --gap).")
    ap.add_argument("--size", default="1600x900", help="Initial window size WxH.")
    args = ap.parse_args()

    w, h = map(int, args.size.lower().split("x"))
    wall = Wall(args.root, args.gap)
    win = "Sighting Wall"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, w, h)
    print(f"[sighting-wall] watching {Path(args.root).resolve()} -- q/Esc to quit, f = fullscreen")
    last = 0.0
    full = False
    while True:
        if time.time() - last >= args.interval:
            wall.poll()
            last = time.time()
        try:
            _, _, ww, wh = cv2.getWindowImageRect(win)
        except cv2.error:
            break
        if ww <= 0 or wh <= 0:
            ww, wh = w, h
        cv2.imshow(win, wall.render(ww, wh))
        key = cv2.waitKey(100) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("f"):
            full = not full
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if full else cv2.WINDOW_NORMAL)
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
