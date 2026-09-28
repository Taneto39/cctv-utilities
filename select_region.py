#!/usr/bin/env python3
"""
select_region.py -- shared polygon-region picker.

Interactive point-and-click UI over a sample frame, saved as region.json.
Used by motion_scan.py (masks motion detection) and censor.py (masks the
blur pass) for the same underlying concept: a fixed area of the camera
frame, picked once per folder, reused across every clip from that camera.

A Region can be made of multiple Shapes (closed polygons) -- press N in
the picker to close the current shape and start a new one. Multiple
Shapes are merged into a single area (one mask, one bounding box); there
is no per-shape threshold or event tracking. See CONTEXT.md. region.json's
`points` is therefore always a list of shapes, each a list of [x, y]
points -- even a single-shape Region is a one-element list. There is no
migration from the old flat single-polygon format; delete and redraw.

Standalone usage (flat folder of clips, e.g. censor.py's input):
    python select_region.py --folder "sound-scan-from-cctv/output"
    # picks the region on the first frame of the first clip (natural-sorted)
    # in that folder, saves <folder>/region.json

    python select_region.py --folder "F:/cam1/output" --frame-source "F:/cam1/output/bad_first_clip.mp4"
    # override which clip's first frame to preview, if the natural first
    # clip is a bad frame (underexposed, motion blur, etc.)

motion_scan.py's own `select-region` subcommand calls the library functions
here directly instead (it grabs its preview frame via ChainedVideoReader,
since its input is <folder>/data rather than a flat folder of clips).
"""
import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".ts", ".dav"}


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def discover_videos_flat(folder):
    """Video files directly inside `folder` (non-recursive), natural-sorted."""
    p = Path(folder)
    files = [f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXTS]
    files.sort(key=lambda f: natural_key(f.name))
    if not files:
        raise FileNotFoundError(f"No video files found in {folder}")
    return files


DRAG_GRAB_RADIUS = 10  # px -- click within this distance of an existing point to drag it
                       # instead of adding a new one (points near the frame edge are fiddly
                       # to place exactly on the first click; this lets you nudge them after)

DISPLAY_MARGIN_W, DISPLAY_MARGIN_H = 60, 120  # px reserved for window chrome/taskbar


def _get_screen_size():
    """Best-effort screen resolution, used to scale down oversized preview
    frames so every corner of the picker window stays on-screen and
    clickable. Falls back to a conservative 1920x1080 guess if it can't be
    determined (e.g. headless/no display server) -- picking still works,
    just without auto-scaling."""
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return w, h
    except Exception:
        return 1920, 1080


def pick_region_ui(frame):
    """Interactive multi-shape polygon picker over `frame`.
    LMB on empty space adds a point to the current shape | LMB-drag an
    existing point (in any shape) moves it | RMB undo (removes the last
    added point of the current shape, or -- once the current shape is
    empty -- drops back to the previous shape) | N closes the current
    shape (needs >= 3 points) and starts a new one | Enter/S save | Esc/Q
    cancel.
    Returns a list of shapes, each a list of (x, y) points in `frame`'s
    own coordinate space -- empty if cancelled or no shape reached 3
    points. Multiple shapes are merged into one Region; see CONTEXT.md.

    `frame` itself is only ever displayed downscaled-to-fit-screen (a plain
    `cv2.namedWindow` renders at the image's native pixel size with no
    scaling, so any frame larger than the screen -- e.g. 2K/4K NVR exports --
    would have corners physically unreachable to click); all picking happens
    in that scaled display space and points are converted back to full-res
    coordinates on return, so callers and region.json's format are
    unaffected."""
    screen_w, screen_h = _get_screen_size()
    scale = min(1.0, (screen_w - DISPLAY_MARGIN_W) / frame.shape[1],
                (screen_h - DISPLAY_MARGIN_H) / frame.shape[0])
    display_frame = (cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                      if scale < 1.0 else frame)

    shapes = [[]]  # shapes[-1] is the in-progress shape; LMB/RMB/N act on it
    dragging = [None]  # (shape_idx, point_idx) of the point being dragged, or None
    window = ("Select Region  [LMB add/drag point | RMB undo | N new shape | "
              "Enter/S save | Esc/Q cancel]")
    max_x, max_y = display_frame.shape[1] - 1, display_frame.shape[0] - 1

    def clamp(x, y):
        return max(0, min(x, max_x)), max(0, min(y, max_y))

    def draw():
        disp = display_frame.copy()
        for s_idx, shape in enumerate(shapes):
            line_color = (255, 0, 0) if s_idx == len(shapes) - 1 else (140, 90, 0)  # dim = finished
            for i, pt in enumerate(shape):
                color = (0, 140, 255) if dragging[0] == (s_idx, i) else (0, 255, 255)
                cv2.circle(disp, pt, 5, color, -1)
                if i > 0:
                    cv2.line(disp, shape[i - 1], pt, line_color, 2)
            if len(shape) > 2:
                cv2.line(disp, shape[-1], shape[0], line_color, 1)
        cv2.imshow(window, disp)

    def nearest_point(x, y):
        best, best_dist2 = None, DRAG_GRAB_RADIUS ** 2
        for s_idx, shape in enumerate(shapes):
            for i, (px, py) in enumerate(shape):
                d2 = (px - x) ** 2 + (py - y) ** 2
                if d2 <= best_dist2:
                    best, best_dist2 = (s_idx, i), d2
        return best

    def on_mouse(event, x, y, flags, param):
        x, y = clamp(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            idx = nearest_point(x, y)
            if idx is not None:
                dragging[0] = idx
            else:
                shapes[-1].append((x, y))
            draw()
        elif event == cv2.EVENT_MOUSEMOVE and dragging[0] is not None:
            s_idx, i = dragging[0]
            shapes[s_idx][i] = (x, y)
            draw()
        elif event == cv2.EVENT_LBUTTONUP:
            dragging[0] = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            if shapes[-1]:
                shapes[-1].pop()
            elif len(shapes) > 1:
                shapes.pop()
            draw()

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)
    draw()
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, ord("s"), ord("S")):
            break
        if key in (27, ord("q"), ord("Q")):
            shapes = [[]]
            break
        if key in (ord("n"), ord("N")) and len(shapes[-1]) >= 3:
            shapes.append([])
            draw()
    cv2.destroyWindow(window)
    shapes = [shape for shape in shapes if len(shape) >= 3]
    if scale < 1.0:
        shapes = [[(round(x / scale), round(y / scale)) for x, y in shape] for shape in shapes]
    return shapes


def save_region(region_path, source, frame, shapes):
    data = {
        "source": str(source),
        "frame_width": frame.shape[1],
        "frame_height": frame.shape[0],
        "points": shapes,
    }
    with open(region_path, "w") as f:
        json.dump(data, f, indent=2)


def load_region(region_path):
    with open(region_path) as f:
        return json.load(f)


def region_mask(shape_hw, points, feather=0):
    """0-255 uint8 mask for `points` (a list of shapes, each a list of
    (x, y) points -- see CONTEXT.md) over an image of size `shape_hw`
    (height, width). Every shape is merged into one mask -- there's no
    per-shape distinction downstream. feather > 0 gaussian-blurs the mask
    edge for a soft, organic boundary instead of a hard cutoff."""
    mask = np.zeros(shape_hw, dtype=np.uint8)
    polys = [np.array(shape, dtype=np.int32) for shape in points]
    cv2.fillPoly(mask, polys, 255)
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), feather)
    return mask


def select_region(input_path, region_path, frame):
    """Run the picker over `frame` and save the result. `input_path` is
    recorded as provenance only -- the caller is responsible for grabbing a
    representative frame (how differs by tool: motion_scan.py chains
    multiple segment files, this module's own CLI just wants the first
    clip's first frame)."""
    shapes = pick_region_ui(frame)
    if not shapes:
        print("Need at least one shape with 3+ points to define a region. Nothing saved.")
        return
    save_region(region_path, input_path, frame, shapes)
    total_points = sum(len(shape) for shape in shapes)
    print(f"Saved region with {len(shapes)} shape(s), {total_points} points total, to {region_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", required=True,
                    help="Folder containing video clips directly (flat, e.g. "
                         "sound-scan-from-cctv/output). region.json is written here "
                         "unless --region overrides it.")
    ap.add_argument("--region", default=None,
                    help="Path to save the region JSON file. Defaults to <folder>/region.json.")
    ap.add_argument("--frame-source", default=None,
                    help="Specific clip to grab the preview frame from, instead of the "
                         "natural-sorted first clip in --folder (use if that clip's first "
                         "frame is a bad preview -- underexposed, motion blur, etc.).")
    args = ap.parse_args()

    folder = Path(args.folder)
    region_path = args.region or str(folder / "region.json")

    if args.frame_source:
        frame_path = Path(args.frame_source)
    else:
        frame_path = discover_videos_flat(folder)[0]

    cap = cv2.VideoCapture(str(frame_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"Could not read a frame from {frame_path}")

    select_region(folder, region_path, frame)


if __name__ == "__main__":
    main()
