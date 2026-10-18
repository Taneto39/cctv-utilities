#!/usr/bin/env python3
"""
nvr_scan.py - Object Scan from an NVR source, and Live Watch. Run through
object_scan.py's subcommands (vocabulary in CONTEXT.md, why vendor SDKs in
docs/adr/0003-nvr-source-uses-vendor-sdks.md):

    # recorded footage, from a time up to the moment the scan was launched:
    python object_scan.py nvr --camera home-hik:15,shop-dahua:1 --from 09:00:00

    # recorded footage, from a time back into the past (newest first), 1h by default:
    python object_scan.py nvr --camera home-hik:15 --from 10:00:00 --direction backward --end 2h

    # live, several cameras, one detector:
    python object_scan.py live --camera home-hik:15,home-hik:3,shop-dahua:1

    # pick a camera's Region on a live snapshot:
    python object_scan.py select-region --camera home-hik:15

    # 2x2 Sighting Wall over every camera folder:
    python sighting_wall.py

Cameras are "<NVR name>:<camera number on the NVR's screen>"; NVR addresses
and passwords live in nvr-sdk/nvrs.env (path to nvr-sdk in this repo's .env).

Every camera gets one folder that accumulates every run (<root>/<nvr>_ch<N>/):
    region.json            optional, from select-region
    downloads.jsonl        one line per downloaded chunk: bytes, seconds, Mbps
    sightings/
      sightings.csv        one row per Sighting, all runs, rebuilt as Hits arrive
      images/              best keyframe per Sighting (boxes + Region outline)
      frames/              every keyframe that had a Hit (images are drawn from these,
                           since the downloaded footage itself is deleted)
      hits_nvr.jsonl       Hits from recorded footage   } Sightings are built from both,
      hits_live.jsonl      Hits from Live Watch         } duplicates counted once
      .nvr_state.json      which stretches of recorded time are already scanned

Recorded footage is downloaded in chunks aligned to a fixed wall-clock grid
(default 5 min), scanned keyframe-by-keyframe, then deleted. Dahua NVRs send
the keyframes only (--full-download turns that off); Hikvision has no such
option, so it downloads every frame.
"""
import argparse
import csv
import heapq
import json
import os
import queue
import re
import struct
import subprocess
import sys
import threading
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

import object_scan as osc
from select_region import load_region, select_region

HERE = Path(__file__).resolve().parent
TIME_FMT = "%Y-%m-%dT%H:%M:%S.%f"
# Footage this recent may not be fully written on the NVR yet: a scan
# waits until a chunk's end is at least this old before downloading it.
SETTLE_S = 90
DOWNLOAD_RETRIES = 3
LIVE_FLUSH_S = 3.0


def _enable_vt(stream):
    """Turn on ANSI escape handling for a Windows console (needed for the
    cursor moves StatusBoard uses); True if the stream can take them."""
    if not stream.isatty():
        return False
    if os.name != "nt":
        return True
    import ctypes
    k32 = ctypes.windll.kernel32
    h = k32.GetStdHandle(-11)   # STD_OUTPUT_HANDLE
    mode = ctypes.c_uint32()
    if not k32.GetConsoleMode(h, ctypes.byref(mode)):
        return False
    return bool(k32.SetConsoleMode(h, mode.value | 0x0004))   # ENABLE_VIRTUAL_TERMINAL_PROCESSING


class StatusBoard:
    """A block of status lines redrawn in place at the bottom of the
    terminal, with log lines printed above it. Replaces one tqdm bar per
    camera: those wrapped or overran the screen with many cameras (every
    redraw moved the cursor by the wrong number of rows, so lines piled up
    on top of each other). Here every line is clipped to the terminal width
    and the block to its height, and every other write -- print(), the SDK
    wrappers' [DBG] lines, warnings -- goes through the board while it's open."""

    def __init__(self, render_lines):
        self.render_lines = render_lines   # () -> [str], first line is the summary
        self.real = sys.stdout
        self.lock = threading.RLock()
        self.drawn = 0
        self.partial = ""
        self.last = None
        self.live = _enable_vt(self.real)
        self.saved = None

    def open(self):
        self.saved = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = _BoardStream(self)
        self.render()
        return self

    def close(self):
        with self.lock:
            if self.saved is None:
                return
            logs = [self.partial] if self.partial else []
            self.partial = ""
            self._paint(logs, force=True)
            if self.live and self.drawn:
                self.real.write("\n")   # leave the final block on screen
                self.real.flush()
            self.drawn = 0
            sys.stdout, sys.stderr = self.saved
            self.saved = None

    def write(self, text):
        with self.lock:
            self.partial += text
            *lines, self.partial = self.partial.split("\n")
            if lines:
                self._paint(lines)

    def log(self, msg):
        self.write(f"{msg}\n")

    def render(self):
        with self.lock:
            self._paint()

    def _paint(self, logs=(), force=False):
        """Log lines (if any) then the block, in one write that overwrites the
        previous block in place. Nothing is erased first -- erasing and then
        redrawing made the whole block flicker -- each line just clears its own
        leftover tail, and an unchanged block isn't redrawn at all."""
        if not self.live or self.saved is None:
            for line in logs:
                self.real.write(line + "\n")
            self.real.flush()
            return
        size = os.get_terminal_size(self.real.fileno())
        width = max(20, size.columns - 1)   # never fill the last column: some consoles wrap there
        lines = self.render_lines()
        room = max(2, size.lines - 2)
        if len(lines) > room:
            hidden = len(lines) - (room - 1)
            lines = lines[:room - 1] + [f"... {hidden} more camera(s) -- widen/heighten the terminal to see them"]
        lines = [line[:width] for line in lines]
        if not logs and not force and lines == self.last and self.drawn:
            return
        out = ["\x1b[?25l\r"]   # hide the cursor while drawing
        if self.drawn > 1:
            out.append(f"\x1b[{self.drawn - 1}A")
        out += [line + "\x1b[K\n" for line in logs]
        out.append("\n".join(line + "\x1b[K" for line in lines))
        out.append("\x1b[J\x1b[?25h")   # drop rows left over if the block shrank
        self.real.write("".join(out))
        self.real.flush()
        self.drawn = len(lines)
        self.last = lines


class _BoardStream:
    """sys.stdout/sys.stderr stand-in that routes writes through a StatusBoard."""

    def __init__(self, board):
        self.board = board

    def write(self, text):
        self.board.write(text)
        return len(text)

    def flush(self):
        pass

    def isatty(self):
        return False   # keep other libraries from drawing their own progress bars

    def __getattr__(self, name):
        return getattr(self.board.real, name)


_board = None
_quiet = threading.Event()   # set on Ctrl+C: download threads still winding down shouldn't print


def _log(*args, **kwargs):
    """print() that goes above the status board (also patched in as the SDK
    wrappers' print, so it takes print()'s arguments)."""
    if _quiet.is_set():
        return
    msg = " ".join(str(a) for a in args)
    if _board is not None:
        _board.log(msg)
    else:
        print(msg, flush=True)


# --------------------------------------------------------------------------
# Config: .env (NVR_SDK_DIR, NVR_SCAN_ROOT) and the shared nvr-sdk
# --------------------------------------------------------------------------

def _dotenv():
    out = {}
    env = HERE / ".env"
    if env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"')
    return out


def _setting(key, default=None):
    return _dotenv().get(key) or os.environ.get(key) or default


def import_nvr():
    sdk_dir = _setting("NVR_SDK_DIR")
    if not sdk_dir or not Path(sdk_dir).is_dir():
        raise SystemExit(f"nvr-sdk not found: set NVR_SDK_DIR in {HERE / '.env'} (see .env.example). "
                         f"Currently: {sdk_dir}")
    if sdk_dir not in sys.path:
        sys.path.insert(0, sdk_dir)
    import nvr
    # the SDK wrappers print() their [DBG] lines; route them around the progress bars
    for mod in ("hik_sdk", "dahua_sdk"):
        try:
            __import__(mod).print = _log
        except Exception:  # SDK not installed / DLL missing: nvr.Camera reports it when used
            pass
    return nvr


def default_root():
    return Path(_setting("NVR_SCAN_ROOT", "nvr_scans"))


# --------------------------------------------------------------------------
# Time arguments
# --------------------------------------------------------------------------

def parse_from(text, now):
    """'HH:MM[:SS]' = today, or a full 'YYYY-mm-dd HH:MM[:SS]'."""
    text = text.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            t = datetime.strptime(text, fmt).time()
            return datetime.combine(now.date(), t)
        except ValueError:
            pass
    raise SystemExit(f"--from '{text}': use HH:MM:SS (today) or 'YYYY-mm-dd HH:MM:SS'.")


def parse_duration(text):
    """'1h', '90m', '1h30m', '45s'; a bare number means minutes."""
    text = text.strip().lower()
    if re.fullmatch(r"\d+(\.\d+)?", text):
        return timedelta(minutes=float(text))
    m = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", text)
    if not m or not any(m.groups()):
        raise SystemExit(f"Duration '{text}': use e.g. 1h, 90m, 1h30m, 45s.")
    h, mi, s = (int(g or 0) for g in m.groups())
    return timedelta(hours=h, minutes=mi, seconds=s)


def plan_chunks(start, end, direction, chunk):
    """[(chunk_start, chunk_end)] covering start..end, cut on a fixed
    wall-clock grid (so the same stretch always gives the same chunks), in
    processing order: oldest first for forward, newest first for backward."""
    step = chunk.total_seconds()
    epoch = datetime(1970, 1, 1)
    out = []
    t = start
    while t < end:
        grid = epoch + timedelta(seconds=(int((t - epoch).total_seconds() // step) + 1) * step)
        ce = min(grid, end)
        out.append((t, ce))
        t = ce
    return out if direction == "forward" else out[::-1]


def _ts(dt):
    return (dt - datetime(1970, 1, 1)).total_seconds()


def _dt(ts):
    return datetime(1970, 1, 1) + timedelta(seconds=ts)


# --------------------------------------------------------------------------
# Per-camera folder: Hits, frames, coverage, Sightings
# --------------------------------------------------------------------------

def _replace(tmp, dst):
    """os.replace that tolerates a reader (the Sighting Wall, Excel) holding
    dst open for a moment -- Windows refuses to replace an open file."""
    for _ in range(50):
        try:
            os.replace(tmp, dst)
            return True
        except PermissionError:
            time.sleep(0.1)
    try:
        os.remove(tmp)
    except OSError:
        pass
    return False


class CameraStore:
    def __init__(self, root, cam_id):
        self.cam_id = cam_id
        self.dir = Path(root) / cam_id
        self.out = self.dir / "sightings"
        self.frames = self.out / "frames"
        self.images = self.out / "images"
        for d in (self.out, self.frames, self.images):
            d.mkdir(parents=True, exist_ok=True)
        region_path = self.dir / "region.json"
        self.region = load_region(region_path) if region_path.exists() else None
        self.state_path = self.out / ".nvr_state.json"
        self.lock = threading.Lock()

    def hits_path(self, src):
        return self.out / f"hits_{src}.jsonl"

    # ---- coverage (recorded footage only) ----

    def load_coverage(self, fp, restart):
        try:
            data = json.loads(self.state_path.read_text())
        except (OSError, json.JSONDecodeError):
            data = None
        self.fp = fp
        if data and data.get("fingerprint") == fp and not restart:
            self.covered = [tuple(iv) for iv in data.get("covered", [])]
            return
        old = self.hits_path("nvr")
        if data and old.exists() and old.stat().st_size:
            # Hit-deciding settings changed (or --restart): earlier Hits answer a
            # different question, so they're set aside rather than mixed in.
            bak = old.with_name(f"hits_nvr.{datetime.now():%Y%m%d-%H%M%S}.bak")
            os.replace(old, bak)
            reason = "--restart" if restart else "Targets/--conf/--imgsz/--model/Region changed"
            print(f"[{self.cam_id}] {reason}: re-scanning; previous Hits moved to {bak.name}")
        self.covered = []
        self._save_coverage()

    def _save_coverage(self):
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"fingerprint": self.fp, "covered": self.covered}))
        _replace(tmp, self.state_path)

    def is_covered(self, cs, ce):
        a, b = _ts(cs), _ts(ce)
        return any(s <= a and b <= e for s, e in self.covered)

    def mark_covered(self, cs, ce):
        ivs = sorted(self.covered + [(_ts(cs), _ts(ce))])
        merged = []
        for s, e in ivs:
            if merged and s <= merged[-1][1] + 0.5:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        self.covered = merged
        self._save_coverage()

    # ---- Hits ----

    def save_frame(self, t, src, frame):
        name = f"{t:%Y%m%d-%H%M%S}.{t.microsecond // 1000:03d}_{src}.jpg"
        cv2.imwrite(str(self.frames / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return name

    def append_hits(self, src, hits):
        with open(self.hits_path(src), "a", encoding="utf-8") as f:
            f.write("".join(json.dumps(h) + "\n" for h in hits))

    def read_hits(self):
        """Hits from every run (recorded + live), each counted once."""
        seen, out = set(), []
        for src in ("nvr", "live"):
            for h in osc.read_hits(self.hits_path(src)):
                t = datetime.strptime(h["t"], TIME_FMT)
                key = (h["target"], h["cls"], round(_ts(t)), tuple(int(v) // 8 for v in h["box"]))
                if key in seen:
                    continue
                seen.add(key)
                h["_t"] = t
                out.append(h)
        return out

    # ---- Sightings ----

    def write_outputs(self, gap):
        with self.lock:
            return self._write_outputs(gap)

    def _write_outputs(self, gap):
        hits = self.read_hits()
        by_target = {}
        for h in hits:
            by_target.setdefault(h["target"], []).append(h)
        sightings = []
        for target, hs in by_target.items():
            hs.sort(key=lambda h: h["_t"])
            cur = None
            for h in hs:
                if cur and (h["_t"] - cur["end"]).total_seconds() <= gap:
                    cur["hits"].append(h)
                    cur["end"] = h["_t"]
                else:
                    cur = {"target": target, "start": h["_t"], "end": h["_t"], "hits": [h]}
                    sightings.append(cur)
        sightings.sort(key=lambda s: (s["start"], s["target"]))

        index_path = self.images / ".index.json"
        try:
            index = json.loads(index_path.read_text())
        except (OSError, json.JSONDecodeError):
            index = {}
        new_index, rows = {}, []
        fmt = "%Y-%m-%d %H:%M:%S"
        for s in sightings:
            hs = s["hits"]
            best = max(hs, key=lambda h: h["conf"])
            name = f"{s['target']}_{s['start']:%Y%m%d-%H%M%S}.jpg"
            if index.get(name) != best["frame"] or not (self.images / name).exists():
                if not self._render(name, best, hs):
                    name = ""
            if name:
                new_index[name] = best["frame"]
            srcs = {h["src"] for h in hs}
            nearest = None
            if self.region:
                nearest = min(osc.region_distance(self.region["points"], (h["box"][0] + h["box"][2]) / 2,
                                                  h["box"][3]) for h in hs)
            rows.append({
                "camera": self.cam_id,
                "target": s["target"],
                "start": s["start"].strftime(fmt),
                "end": s["end"].strftime(fmt),
                "duration_s": round((s["end"] - s["start"]).total_seconds(), 1),
                "hits": len({h["frame"] for h in hs}),
                "max_conf": best["conf"],
                "raw_classes": ";".join(f"{c}:{n}" for c, n in Counter(h["cls"] for h in hs).most_common()),
                "nearest_px": "" if nearest is None else round(nearest),
                "source": "live" if srcs == {"live"} else "nvr" if srcs == {"nvr"} else "nvr+live",
                "image": name,
            })
        for stale in set(index) - set(new_index):
            try:
                (self.images / stale).unlink()
            except OSError:
                pass
        tmp = index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(new_index, indent=1))
        _replace(tmp, index_path)
        csv_path = self.out / "sightings.csv"
        tmp = csv_path.with_suffix(".tmp")
        fields = ["camera", "target", "start", "end", "duration_s", "hits", "max_conf",
                  "raw_classes", "nearest_px", "source", "image"]
        with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        _replace(tmp, csv_path)
        return sightings

    def _render(self, name, best, hits):
        frame = cv2.imread(str(self.frames / best["frame"]))
        if frame is None:
            return False
        if self.region:
            for shape in self.region["points"]:
                cv2.polylines(frame, [np.array(shape, dtype=np.int32)], True, (0, 255, 255), 1)
        for h in hits:
            if h["frame"] != best["frame"]:
                continue
            x1, y1, x2, y2 = map(int, h["box"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{h['cls']} {h['conf']:.2f}"
            cv2.putText(frame, label, (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(frame, label, (x1, max(12, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        tmp = self.images / (name + ".tmp.jpg")
        cv2.imwrite(str(tmp), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return _replace(tmp, self.images / name)


# --------------------------------------------------------------------------
# Detector: one model for every camera, fed from one queue
# --------------------------------------------------------------------------
# Queue items:
#   ("frame", cam_id, wallclock_datetime, bgr_frame, src)
#   ("mark", cam_id, callback)    -- runs after every earlier frame's Hits are written
#   ("done", cam_id)              -- that camera's producer has finished

class Detector:
    def __init__(self, args, stores):
        self.model = osc.load_model(args.model, args.batch_size, use_trt=not args.no_trt)
        self.class_targets = osc.resolve_targets(self.model.names, args.targets)
        self.class_ids = sorted(self.class_targets)
        self.conf = args.conf
        self.batch_size = args.batch_size
        self.stores = stores
        self.crops = {}
        self.imgsz = {}
        self.n_frames = 0
        self.n_hits = Counter()
        self.last_t = {}     # cam_id -> wall-clock time of the latest keyframe run through the detector
        self.on_hits = None
        for cam_id, store in stores.items():
            self.imgsz[cam_id] = args.imgsz or (640 if store.region else 1280)
        # one predict() sets up ultralytics' predictor (conf/classes included);
        # after that _prepare/_finish drive its stages directly
        self.model.predict(np.zeros((64, 64, 3), np.uint8), imgsz=640, conf=self.conf,
                           classes=self.class_ids, verbose=False)
        self.predictor = self.model.predictor
        # an end-to-end model (yolo26) outputs a small (batch, 300, 6) tensor whose
        # "NMS" is only a few masks per image; on the GPU each one is a sync point
        self.cpu_post = bool(getattr(self.predictor.model, "end2end", False))
        self.prep_pool = ThreadPoolExecutor(max_workers=1)
        self.pending = None   # Future of the next batch being prepared

    def fingerprint(self, cam_id, model_path):
        st = self.stores[cam_id]
        return osc.fingerprint(self.class_targets, self.conf, self.imgsz[cam_id], model_path, st.region)

    def _crop(self, cam_id, frame):
        store = self.stores[cam_id]
        if not store.region:
            return None
        h, w = frame.shape[:2]
        key = (cam_id, w, h)  # per size: a camera's resolution can change mid-scan
        if key not in self.crops:
            self.crops[key] = osc.region_crop_box(store.region["points"], w, h)
        return self.crops[key]

    # Batches run in two stages so the GPU never waits on the CPU: _prepare
    # (crop, letterbox, upload; ~4.6ms/image) runs on prep_pool for batch k+1
    # while _finish (TensorRT inference ~5.7ms/image, NMS, Hits) runs batch k.
    # In one predict() call those ran back to back and the GPU idled ~45% of the
    # time: ~95 -> ~150 images/s, boxes identical to predict() (DESIGN.md
    # "Overlapped preprocessing").

    def _submit(self, batch):
        nxt = self.prep_pool.submit(self._prepare, batch)
        self.flush()
        self.pending = nxt

    def flush(self):
        """Finish the batch being prepared, if any."""
        if self.pending is not None:
            groups, n = self.pending.result()
            self.pending = None
            self._finish(groups, n)

    def _letterbox(self, imgs, imgsz):
        """predict()'s preprocessing, bit for bit: the same LetterBox, then
        BGR->RGB, HWC->CHW and /255 done on the GPU instead of in numpy."""
        import torch  # deferred like ultralytics in object_scan.py: --help shouldn't need them
        from ultralytics.data.augment import LetterBox
        from ultralytics.utils.checks import check_imgsz
        p = self.predictor
        sz = check_imgsz(imgsz, stride=p.model.stride, min_dim=2)
        auto = (len({x.shape for x in imgs}) == 1 and p.args.rect
                and (p.model.format == "pt" or (getattr(p.model, "dynamic", False) and p.model.format != "imx")))
        lb = LetterBox(sz, auto=auto, stride=p.model.stride)
        t = torch.from_numpy(np.stack([lb(image=x) for x in imgs])).to(p.device)
        t = t.flip(-1).permute(0, 3, 1, 2).contiguous()
        t = t.half() if p.model.fp16 else t.float()
        return t / 255

    def _prepare(self, batch):
        # one inference per (imgsz, input size): ultralytics letterboxes a batch of
        # mixed sizes to a square instead of each image's own rectangle, which
        # changes detections (a 0.40 Hit vanished when a 2688x1520 frame shared a
        # batch with 2560x1440 ones) -- so cameras of different resolutions, or
        # with / without a Region, never share a batch
        groups = {}
        for item in batch:
            _, cam_id, t, frame, src = item
            crop = self._crop(cam_id, frame)
            if crop:
                x0, y0, x1, y1 = crop
                img, off = frame[y0:y1, x0:x1], (x0, y0)
            else:
                img, off = frame, (0, 0)
            if img.size == 0:
                # an empty image makes ultralytics' letterbox divide by zero and
                # kills the whole run -- drop the frame instead
                _log(f"[{cam_id}] skipped an empty {frame.shape[1]}x{frame.shape[0]} frame at {t:%m-%d %H:%M:%S}")
                continue
            g = groups.setdefault((self.imgsz[cam_id], img.shape[:2]), ([], [], []))
            g[0].append(item)
            g[1].append(img)
            g[2].append(off)
        return [(items, imgs, offs, self._letterbox(imgs, imgsz))
                for (imgsz, _), (items, imgs, offs) in groups.items()], len(batch)

    def _finish(self, groups, n):
        p = self.predictor
        for items, imgs, offs, tensor in groups:
            p.batch = ([""] * len(imgs), imgs, [""] * len(imgs))   # postprocess reads paths from it
            preds = p.inference(tensor)
            if self.cpu_post:
                # one copy, then postprocess on the CPU: bit-identical boxes, and the
                # main thread no longer waits on a GPU sync per mask (DESIGN.md
                # "Postprocess on the CPU")
                preds = (preds[0] if isinstance(preds, (list, tuple)) else preds).cpu()
            results = p.postprocess(preds, tensor, imgs)
            for (_, cam_id, t, frame, src), (ox, oy), r in zip(items, offs, results):
                self.last_t[cam_id] = t
                if not len(r.boxes):
                    continue
                store = self.stores[cam_id]
                fname = store.save_frame(t, src, frame)
                hits = []
                for b in r.boxes:
                    cls = int(b.cls)
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    box = [round(x1 + ox, 1), round(y1 + oy, 1), round(x2 + ox, 1), round(y2 + oy, 1)]
                    for target in self.class_targets[cls]:
                        hits.append({"t": t.strftime(TIME_FMT)[:-3], "target": target, "cls": self.model.names[cls],
                                     "conf": round(float(b.conf), 3), "box": box, "frame": fname, "src": src})
                store.append_hits(src, hits)
                self.n_hits[cam_id] += len(hits)
                if self.on_hits:
                    self.on_hits(cam_id)
        self.n_frames += n

    def run(self, q, n_producers, on_hits=None, idle=None, tick=None):
        """Consumes q until every producer has sent "done". `idle` (if given)
        is called whenever the queue has been empty for a moment; `tick`
        (if given) after every item and when idle."""
        self.on_hits = on_hits
        remaining = n_producers
        batch = []
        while remaining > 0:
            if tick:
                tick()
            try:
                item = q.get(timeout=0.5)
            except queue.Empty:
                if batch:
                    self._submit(batch)
                    batch = []
                self.flush()
                if idle:
                    idle()
                continue
            kind = item[0]
            if kind == "frame":
                batch.append(item)
                if len(batch) >= self.batch_size:
                    self._submit(batch)
                    batch = []
                continue
            if batch:
                self._submit(batch)
                batch = []
            self.flush()   # a mark runs only after every earlier frame's Hits are written
            if kind == "mark":
                item[2]()
            elif kind == "done":
                remaining -= 1


# --------------------------------------------------------------------------
# Recorded footage (Object Scan, NVR source)
# --------------------------------------------------------------------------

class FairGate:
    """Caps concurrent downloads on one NVR, handing a free slot to whichever
    waiting camera is furthest behind in the scan (lowest priority value)
    instead of whichever thread happens to grab it first. A plain semaphore
    let some cameras on a busy NVR re-take slots over and over while others
    starved for hours; this keeps every camera on an NVR moving together."""

    def __init__(self, n):
        self.n, self.busy = n, 0
        self.cv = threading.Condition()
        self.waiting = []   # heap of (priority, seq)
        self.seq = 0

    def acquire(self, priority):
        with self.cv:
            self.seq += 1
            entry = (priority, self.seq)
            heapq.heappush(self.waiting, entry)
            while self.busy >= self.n or self.waiting[0] != entry:
                self.cv.wait()
            heapq.heappop(self.waiting)
            self.busy += 1
            self.cv.notify_all()   # the next in line may fit too

    def release(self):
        with self.cv:
            self.busy -= 1
            self.cv.notify_all()


def _vram_reader():
    """A function returning the GPU's (used, total) VRAM bytes (all
    processes, not just this one), or None when it can't be read. NVML
    (nvidia-ml-py) sees the ffmpeg decoders' memory too; torch's mem_get_info
    is the fallback."""
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)

        def read():
            m = pynvml.nvmlDeviceGetMemoryInfo(h)
            return m.used, m.total
        read()
        return read
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            def read():
                free, total = torch.cuda.mem_get_info()
                return total - free, total
            read()
            return read
    except Exception:
        pass
    return lambda: None


class DecoderGate:
    """Caps concurrent keyframe decoders (one ffmpeg + NVDEC session each)
    across every camera, sized automatically. With one decoder thread per
    camera and the detector as the bottleneck, every camera's ffmpeg sat
    blocked on the full frame queue while still holding its CUDA context,
    NVDEC surfaces and ~0.5GB of RAM: 31 of them at 32 cameras filled VRAM,
    pushed the detector into shared GPU memory and cut it from ~90 to 19
    keyframes/s (DESIGN.md "Decoder gate"). A new decoder starts only while
    the detector is short of frames (queue under half full) and VRAM is under
    `vram_limit`; one always may, so nothing deadlocks. Like FairGate, the
    camera furthest behind goes first."""

    # A just-started ffmpeg takes ~1s to allocate its VRAM, so decoders started
    # in the last SETTLE_S count as DECODER_VRAM each on top of the reading.
    # (A plain "wait SETTLE_S after each start" capped it at one decoder: a
    # chunk decodes in ~2s, so every replacement reset the wait.)
    SETTLE_S = 2.0
    DECODER_VRAM = 512 * 2**20

    def __init__(self, q, vram_limit):
        self.q, self.limit = q, vram_limit
        self.read_vram = _vram_reader()
        self.busy = self.peak = 0
        self.starts = deque()   # start times of recent decoders
        self.cv = threading.Condition()
        self.waiting = []   # heap of (priority, seq)
        self.seq = 0

    def _may_start(self):
        if self.busy == 0:
            return True
        if self.q.qsize() >= self.q.maxsize // 2:
            return False
        vram = self.read_vram()
        if vram is None:
            return True
        now = time.time()
        while self.starts and now - self.starts[0] > self.SETTLE_S:
            self.starts.popleft()
        used, total = vram
        return used + len(self.starts) * self.DECODER_VRAM < self.limit * total

    def acquire(self, priority):
        with self.cv:
            self.seq += 1
            entry = (priority, self.seq)
            heapq.heappush(self.waiting, entry)
            # timed wait: queue depth and VRAM change without anyone notifying
            while self.waiting[0] != entry or not self._may_start():
                self.cv.wait(timeout=0.5)
            heapq.heappop(self.waiting)
            self.busy += 1
            self.peak = max(self.peak, self.busy)
            self.starts.append(time.time())
            self.cv.notify_all()

    def release(self):
        with self.cv:
            self.busy -= 1
            self.cv.notify_all()

    def summary(self):
        # decoders running (+ chunks downloaded and waiting for one), frames
        # queued for the detector: a full queue = detector-bound, an empty one
        # with nothing waiting = download-bound
        vram = self.read_vram()
        wait = f"+{len(self.waiting)}w" if self.waiting else ""
        return (f"dec {self.busy}{wait}, q {self.q.qsize()}/{self.q.maxsize}"
                + (f", VRAM {vram[0] / vram[1]:.0%}" if vram is not None else ""))


class DownloadStats:
    """Download throughput of one NVR, for the status board: the overall
    average since the first download, and the per-connection speed of the
    last few chunks (a single SDK download can be capped, see DESIGN.md)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.t0 = None
        self.bytes = 0
        self.recent = deque(maxlen=10)   # Mbps of the last chunks, one connection each

    def started(self):
        with self.lock:
            if self.t0 is None:
                self.t0 = time.time()

    def add(self, nbytes, secs):
        with self.lock:
            self.bytes += nbytes
            if secs > 0:
                self.recent.append(nbytes * 8 / secs / 1e6)

    def summary(self):
        with self.lock:
            if self.t0 is None or not self.recent:
                return "speed ?"
            total = self.bytes * 8 / max(time.time() - self.t0, 1e-6) / 1e6
            per = sum(self.recent) / len(self.recent)
            return f"{total:.0f} Mbps avg, {per:.1f}/conn"


class NvrCameraJob:
    """Downloads one camera's chunks (a little ahead) and decodes their
    keyframes into the shared detector queue."""

    def __init__(self, cam, store, chunks, q, args, stats, gate, speed, decoders):
        self.cam, self.store, self.chunks, self.q = cam, store, chunks, q
        self.args, self.stats, self.gate, self.speed = args, stats, gate, speed
        self.decoders = decoders              # DecoderGate shared by every camera
        self.status = {"state": "starting"}   # set by the download threads, shown on the progress bar
        self.active = {}                      # chunk start -> its download state
        self.active_lock = threading.Lock()
        self.log_lock = threading.Lock()      # downloads.jsonl, appended by several download threads
        self.bar = None
        self.on_progress = None
        self.ready = queue.Queue()
        # chunks in flight or downloaded but not yet decoded; the decoder frees one
        # per chunk, so at most this many chunk files sit in .chunks at once
        self.ahead = threading.Semaphore(args.prefetch + 1)
        self.tmp = store.dir / ".chunks"
        self.tmp.mkdir(exist_ok=True)
        # half-downloaded chunks left by a run stopped with Ctrl+C
        for leftover in self.tmp.iterdir():
            try:
                leftover.unlink()
            except OSError:
                pass
        # Dahua can send I-frames only (DESIGN.md "Keyframe-only download"):
        # same keyframes, ~2x fewer bytes, exact wall-clock time per frame
        self.keyframes = getattr(cam, "supports_keyframes", False) and not args.full_download

    def start(self):
        threading.Thread(target=self._download_all, daemon=True).start()
        threading.Thread(target=self._decode_all, daemon=True).start()

    def _set_state(self, cs, text):
        """Record one chunk's download state (None = finished) and show the
        earliest active chunk's on the progress bar."""
        with self.active_lock:
            if text is None:
                self.active.pop(cs, None)
            else:
                self.active[cs] = text
            if self.active:
                first = min(self.active)
                more = f" (+{len(self.active) - 1} more)" if len(self.active) > 1 else ""
                self.status["state"] = self.active[first] + more
            else:
                self.status["state"] = "downloaded ahead, waiting for detector"

    def short_state(self):
        """The download state in a few characters, so a camera fits on one
        status-board line: counts of chunks downloading (D), waiting for an NVR
        slot (w), retrying (!) and waiting for the NVR to finish recording (rec),
        then the start of the next chunk in scan order; R = all downloaded chunks are
        waiting for the detector, "all dl" = nothing left to download."""
        with self.active_lock:
            active = dict(self.active)
        if not active:
            state = self.status["state"]
            return "all dl" if state == "all downloaded" else ("starting" if state == "starting" else "R")
        counts = Counter()
        for s in active.values():
            if s.startswith(("downloading", "keyframes")):
                counts["D"] += 1
            elif s.startswith("waiting for NVR slot"):
                counts["w"] += 1
            elif s.startswith("retrying"):
                counts["!"] += 1
            else:
                counts["rec"] += 1
        parts = [f"{k}{counts[k]}" for k in ("D", "w", "!", "rec") if counts[k]]
        nxt = min(active) if self.args.direction == "forward" else max(active)
        return " ".join(parts) + f" @{nxt:%H:%M}"

    def _download_all(self):
        # Several chunks of one camera download at once (up to --prefetch + 1),
        # since some NVRs cap each connection's speed (a Dahua NVR measured
        # ~21Mbps per connection, adding up linearly across connections) -- one chunk at
        # a time per camera left the NVR's --connections slots idle when there
        # were fewer cameras than slots. Chunks still reach the decoder in order.
        with ThreadPoolExecutor(max_workers=self.args.prefetch + 1) as pool:
            pending = deque()
            for cs, ce in self.chunks:
                # a slot is freed by the decoder once a chunk is scanned; if every
                # slot is held by a chunk not handed over yet, hand over the oldest
                while not self.ahead.acquire(blocking=False):
                    if not pending:
                        self.ahead.acquire()
                        break
                    self.ready.put(pending.popleft().result())
                pending.append(pool.submit(self._fetch_chunk, cs, ce))
                while pending and pending[0].done():
                    self.ready.put(pending.popleft().result())
            while pending:
                self.ready.put(pending.popleft().result())
        self.status["state"] = "all downloaded"
        self.ready.put(None)

    def _fetch_chunk(self, cs, ce):
        ext = ".dav" if self.cam.cfg["vendor"] == "dahua" else ".mp4"
        # either direction: a chunk ending near "now" may not be fully recorded yet
        wait = _ts(ce) + SETTLE_S - _ts(datetime.now())
        if wait > 0:
            self._set_state(cs, f"waiting for NVR to finish recording {ce:%H:%M}")
            time.sleep(wait)
        dst = self.tmp / f"{cs:%Y%m%d-%H%M%S}{ext}"
        what = "keyframes" if self.keyframes else "downloading"
        fetch = self.cam.download_keyframes if self.keyframes else self.cam.download
        # position in scan order: the camera furthest behind gets the next slot
        priority = _ts(cs) if self.args.direction == "forward" else -_ts(cs)
        result = False
        for attempt in range(DOWNLOAD_RETRIES + 1):
            self._set_state(cs, f"waiting for NVR slot {cs:%H:%M}-{ce:%H:%M}")
            self.gate.acquire(priority)
            self._set_state(cs, f"{what} {cs:%H:%M}-{ce:%H:%M}")
            self.speed.started()
            t = time.time()
            try:
                result = fetch(cs, ce, str(dst), timeout=self.args.chunk_timeout)
            except Exception as e:  # SDK hiccup: retry like a failed download
                _log(f"[{self.store.cam_id}] download {cs:%H:%M:%S} error: {e}")
                result = False
            finally:
                self.gate.release()
            if result:
                self._record_download(cs, ce, dst, time.time() - t, attempt)
            if result is not False:
                break
            if attempt < DOWNLOAD_RETRIES:
                self._set_state(cs, f"retrying {cs:%H:%M} ({attempt + 1}/{DOWNLOAD_RETRIES})")
                time.sleep(5 * 3 ** attempt)
        # a keyframe-only file starts at the first keyframe >= cs, not at cs
        base = cs
        if result and self.keyframes:
            import dahua_sdk  # on sys.path once nvr is imported
            try:
                with open(dst, "rb") as f:
                    base = dahua_sdk.dhav_time(f.read(24))
            except (OSError, ValueError, struct.error) as e:
                _log(f"[{self.store.cam_id}] keyframe time {cs:%H:%M:%S} unreadable ({e}), using chunk start")
        self._set_state(cs, None)
        return cs, ce, dst if result else None, result, base

    def _record_download(self, cs, ce, dst, secs, attempt):
        """Feed one finished download into the NVR's speed and append it to
        <camera>/downloads.jsonl, to tell an NVR cap from a network one later."""
        try:
            nbytes = os.path.getsize(dst)
        except OSError:
            return
        self.speed.add(nbytes, secs)
        rec = {"time": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "nvr": self.cam.nvr_name,
               "chunk": f"{cs:%Y-%m-%d %H:%M:%S}", "chunk_s": (ce - cs).total_seconds(),
               "mode": "keyframes" if self.keyframes else "full", "bytes": nbytes,
               "secs": round(secs, 2), "mbps": round(nbytes * 8 / max(secs, 1e-6) / 1e6, 2),
               "slots_busy": self.gate.busy, "attempt": attempt}
        try:
            with self.log_lock, open(self.store.dir / "downloads.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError:
            pass

    def _decode_all(self):
        cam_id = self.store.cam_id
        try:
            while True:
                item = self.ready.get()
                if item is None:
                    break
                cs, ce, path, result, base = item
                ok = True
                if path is not None:
                    self.decoders.acquire(_ts(cs) if self.args.direction == "forward" else -_ts(cs))
                    try:
                        for offset, frame in osc.iter_keyframes(path):
                            self.q.put(("frame", cam_id, base + timedelta(seconds=offset), frame, "nvr"))
                    except Exception as e:
                        _log(f"[{cam_id}] decode {cs:%H:%M:%S} failed: {e}")
                        ok = False
                    finally:
                        self.decoders.release()
                        try:
                            os.remove(path)
                        except OSError:
                            pass
                self.q.put(("mark", cam_id, lambda cs=cs, ce=ce, r=result, ok=ok: self._chunk_done(cs, ce, r, ok)))
                self.ahead.release()
        finally:
            self.q.put(("done", cam_id))

    def _chunk_done(self, cs, ce, result, ok):
        """Runs on the detector thread, after this chunk's Hits are written."""
        st = self.stats[self.store.cam_id]
        if result is None:
            st["no_footage"] += 1
            self.store.mark_covered(cs, ce)
        elif result and ok:
            st["ok"] += 1
            self.store.mark_covered(cs, ce)
        else:
            st["failed"].append(f"{cs:%H:%M:%S}-{ce:%H:%M:%S}")
        if self.on_progress:
            self.on_progress((ce - cs).total_seconds())
        self.store.write_outputs(self.args.gap)


def _close_cameras(cams, timeout=None):
    """Log every camera out of its NVR. With a timeout (Ctrl+C), give up
    waiting after that many seconds: an SDK logout can block while a
    download on the same login is still running."""
    def close_all():
        for cam in cams:
            try:
                cam.close()
            except Exception:
                pass
    t = threading.Thread(target=close_all, daemon=True)
    t.start()
    t.join(timeout)


def run_nvr(args):
    nvr = import_nvr()
    now = datetime.now().replace(microsecond=0)
    start = parse_from(args.start, now)
    if args.direction == "forward":
        if start >= now:
            raise SystemExit(f"--from {start} is in the future (now {now}).")
        lo, hi = start, now
    else:
        hi = min(start, now)
        lo = hi - parse_duration(args.end)
    chunks_all = plan_chunks(lo, hi, args.direction, timedelta(minutes=args.chunk_min))

    nvrs = nvr.load_nvrs()
    root = Path(args.root)
    cams, stores = {}, {}
    for name, no in nvr.parse_cameras(args.camera):
        cam_id = f"{name}_ch{no}"
        try:
            # login now: the SDK login is lazy, and a bad password should skip this
            # camera here rather than fail every chunk inside the download threads
            cams[cam_id] = nvr.Camera(name, no, max_connections=args.connections, nvrs=nvrs).login()
        except (RuntimeError, SystemExit) as e:
            print(f"[{cam_id}] skipped: {e}")
            continue
        stores[cam_id] = CameraStore(root, cam_id)
    if not cams:
        raise SystemExit("No camera could be opened.")

    det = Detector(args, stores)
    q = queue.Queue(maxsize=args.batch_size * 8)
    decoders = DecoderGate(q, args.vram_limit / 100)
    stats = {c: {"ok": 0, "no_footage": 0, "failed": [], "skipped": 0} for c in cams}
    jobs, total_s = [], 0.0
    gates, speeds = {}, {}   # one each per NVR, shared by its cameras
    for cam_id, cam in cams.items():
        store = stores[cam_id]
        store.load_coverage(det.fingerprint(cam_id, args.model), args.restart)
        todo = [(cs, ce) for cs, ce in chunks_all if not store.is_covered(cs, ce)]
        stats[cam_id]["skipped"] = len(chunks_all) - len(todo)
        total_s += sum((ce - cs).total_seconds() for cs, ce in todo)
        if todo:
            gate = gates.setdefault(cam.nvr_name, FairGate(args.connections))
            speed = speeds.setdefault(cam.nvr_name, DownloadStats())
            jobs.append((cam_id, NvrCameraJob(cam, store, todo, q, args, stats, gate, speed, decoders)))
    print(f"[object-scan nvr] {len(cams)} camera(s), {args.direction} {lo:%Y-%m-%d %H:%M:%S} -> "
          f"{hi:%Y-%m-%d %H:%M:%S}, {len(chunks_all)} chunk(s) of {args.chunk_min} min each per camera"
          + "".join(f"\n  {c}: {stats[c]['skipped']} chunk(s) already scanned" for c in cams if stats[c]["skipped"]))

    t0 = time.time()
    done = {"all": 0.0}
    for cam_id, job in jobs:
        job.total_s = sum((ce - cs).total_seconds() for cs, ce in job.chunks)
        done[cam_id] = 0.0

    def progress(cam_id, sec):
        done[cam_id] += sec
        done["all"] += sec

    by_nvr = {}   # NVR name -> [(cam_id, job)], in camera order
    for cam_id, job in jobs:
        by_nvr.setdefault(job.cam.nvr_name, []).append((cam_id, job))
    width = max([len(c) for c in cams] + [3])

    def bar(frac, n=20):
        k = int(round(max(0.0, min(1.0, frac)) * n))
        return "#" * k + "-" * (n - k)

    def hms(s):
        s = int(s)
        return f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}"

    rate_hist = deque()   # (time, det.n_frames) samples for the recent keyframe rate

    def status_lines():
        # one line for everything, then per NVR: a summary line (progress, how far
        # into the footage its slowest camera is, slots, download speed) and its
        # cameras packed several to a line -- one line per camera filled the screen
        now = time.time()
        elapsed = now - t0
        frac = done["all"] / total_s if total_s else 1.0
        eta = hms(elapsed * (total_s - done["all"]) / done["all"]) if done["all"] else "?"
        # keyframes/s over the last ~30s: the since-start average hides the
        # slow start-up (first downloads) and any later change
        rate_hist.append((now, det.n_frames))
        while len(rate_hist) > 2 and now - rate_hist[1][0] >= 30:
            rate_hist.popleft()
        t_old, n_old = rate_hist[0]
        recent = (det.n_frames - n_old) / max(now - t_old, 1e-6)
        lines = [f"{'all'.ljust(width)} [{bar(frac, 10)}] {frac:4.0%} | {hms(elapsed)}, ~{eta} left "
                 f"| {det.n_frames} kf, {recent:.0f}/s | {decoders.summary()} "
                 f"| hits {sum(det.n_hits.values())}"]
        for name, members in by_nvr.items():
            # per NVR: slots in use, downloads queued for one, download speed
            gate = gates[name]
            if all(done[c] >= job.total_s for c, job in members):
                lines.append(f"{name}: done")
            else:
                lines.append(f"{name}: slots {gate.busy}/{gate.n}, {len(gate.waiting)} queued | "
                             f"{speeds[name].summary()}")
            # then one short line per camera, so it fits without being cut off
            for cam_id, job in members:
                frac = done[cam_id] / job.total_s if job.total_s else 1.0
                at = det.last_t.get(cam_id)
                state = "done" if frac >= 1 else job.short_state()
                lines.append(f"{cam_id.ljust(width)} [{bar(frac, 10)}] {frac:4.0%} | "
                             f"{f'{at:%m-%d %H:%M:%S}' if at else '-':14} | hits {det.n_hits[cam_id]:<4} | {state}")
        lines.append("D downloading, w slot wait, ! retry, rec still recording, @ next chunk, R detector wait")
        return lines

    global _board
    _board = StatusBoard(status_lines)
    interrupted = False
    try:
        _board.open()
        for cam_id, job in jobs:
            job.on_progress = (lambda c: lambda sec: progress(c, sec))(cam_id)
            job.start()
        last_tick = [0.0]

        def tick(force=False):
            if not force and time.time() - last_tick[0] < 0.5:
                return
            last_tick[0] = time.time()
            _board.render()

        # Sightings are rebuilt after every chunk anyway; also rebuild (throttled)
        # as soon as a Hit lands, so the Sighting Wall shows it without
        # waiting for the rest of the chunk
        dirty, last = set(), [0.0]

        def flush():
            if dirty and time.time() - last[0] >= LIVE_FLUSH_S:
                for cam_id in list(dirty):
                    stores[cam_id].write_outputs(args.gap)
                dirty.clear()
                last[0] = time.time()

        det.run(q, len(jobs), on_hits=lambda cam_id: (dirty.add(cam_id), flush()), idle=flush, tick=tick)
        tick(force=True)
    except KeyboardInterrupt:
        interrupted = True
        # downloads still in flight fail once their camera logs out below; their
        # [DBG]/retry lines would only bury the summary
        _quiet.set()
    finally:
        _board.close()
        _board = None
        _close_cameras(cams.values(), timeout=5 if interrupted else None)
    if interrupted:
        print("\n[object-scan nvr] stopped (Ctrl+C). Finished chunks are saved -- run the same command "
              "again to continue from where it stopped.")
        for cam_id, store in stores.items():
            store.write_outputs(args.gap)
            print(f"  {cam_id}: {stats[cam_id]['ok']} chunk(s) scanned this run")
        sys.stdout.flush()
        # skip joining the download threads: each would sit in its SDK call
        # until the chunk finishes or times out (--chunk-timeout, minutes)
        os._exit(130)
    elapsed = time.time() - t0
    if det.n_frames:
        print(f"[object-scan nvr] {det.n_frames} keyframes in {elapsed:.0f}s "
              f"({total_s / max(elapsed, 1e-6):.0f}x real time over all cameras, up to {decoders.peak} decoder(s) at once)")
    for cam_id, store in stores.items():
        sightings = store.write_outputs(args.gap)
        st = stats[cam_id]
        in_range = [s for s in sightings if s["end"] >= lo and s["start"] <= hi]
        line = (f"  {cam_id}: {st['ok']} chunk(s) scanned, {st['no_footage']} with no footage, "
                f"{len(in_range)} Sighting(s) in range -> {store.out / 'sightings.csv'}")
        if st["failed"]:
            line += f"\n    FAILED (not marked scanned, re-run to retry): {', '.join(st['failed'])}"
        print(line)


# --------------------------------------------------------------------------
# Live Watch
# --------------------------------------------------------------------------

RTSP_IN = ["-rtsp_transport", "tcp"]


def _live_reader(cam, cam_id, q, stop):
    backoff = 5
    while not stop.is_set():
        t_start = None
        got = 0
        try:
            for offset, frame in osc.iter_keyframes(cam.rtsp_url(), input_args=RTSP_IN):
                if stop.is_set():
                    break
                if t_start is None:
                    # wall clock of the stream's first keyframe; later ones keep the
                    # stream's own spacing instead of when they happened to arrive
                    t_start = datetime.now() - timedelta(seconds=offset)
                    print(f"[{cam_id}] live: connected")
                    backoff = 5
                got += 1
                q.put(("frame", cam_id, t_start + timedelta(seconds=offset), frame, "live"))
        except Exception as e:
            print(f"[{cam_id}] live: {type(e).__name__}: {str(e)[:200]}")
        if stop.is_set():
            break
        print(f"[{cam_id}] live: stream ended after {got} keyframe(s), reconnecting in {backoff}s")
        stop.wait(backoff)
        backoff = min(backoff * 2, 60)
    q.put(("done", cam_id))


def run_live(args):
    nvr = import_nvr()
    nvrs = nvr.load_nvrs()
    root = Path(args.root)
    cams, stores = {}, {}
    for name, no in nvr.parse_cameras(args.camera):
        cam_id = f"{name}_ch{no}"
        if name not in nvrs:
            print(f"[{cam_id}] skipped: no NVR named '{name}' in nvrs.env")
            continue
        cams[cam_id] = nvr.Camera(name, no, nvrs=nvrs)   # RTSP only: no SDK login
        stores[cam_id] = CameraStore(root, cam_id)
    if not cams:
        raise SystemExit("No camera to watch.")
    det = Detector(args, stores)
    q = queue.Queue(maxsize=args.batch_size * 8)
    stop = threading.Event()
    dirty, last_flush = set(), [time.time()]

    def flush(force=False):
        if dirty and (force or time.time() - last_flush[0] >= LIVE_FLUSH_S):
            for cam_id in list(dirty):
                stores[cam_id].write_outputs(args.gap)
            dirty.clear()
            last_flush[0] = time.time()

    def on_hits(cam_id):
        dirty.add(cam_id)
        flush()

    for cam_id, cam in cams.items():
        threading.Thread(target=_live_reader, args=(cam, cam_id, q, stop), daemon=True).start()
    print(f"[object-scan live] watching {len(cams)} camera(s): {', '.join(cams)}. Ctrl+C to stop.")
    try:
        det.run(q, len(cams), on_hits=on_hits, idle=flush)
    except KeyboardInterrupt:
        print("\n[object-scan live] stopping...")
        stop.set()
    finally:
        flush(force=True)
    for cam_id, n in det.n_hits.items():
        print(f"  {cam_id}: {n} Hit(s) this session -> {stores[cam_id].out / 'sightings.csv'}")


# --------------------------------------------------------------------------
# select-region on a live snapshot
# --------------------------------------------------------------------------

def run_select_region(args):
    nvr = import_nvr()
    nvrs = nvr.load_nvrs()
    (name, no), = nvr.parse_cameras(args.camera)
    if name not in nvrs:
        raise SystemExit(f"No NVR named '{name}' in nvrs.env")
    cam = nvr.Camera(name, no, nvrs=nvrs)   # RTSP only: no SDK login
    cam_dir = Path(args.root) / f"{name}_ch{no}"
    cam_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", *RTSP_IN, "-i", cam.rtsp_url(), "-frames:v", "1",
           "-f", "image2pipe", "-vcodec", "bmp", "-"]
    out = subprocess.run(cmd, capture_output=True, timeout=30).stdout
    frame = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR) if out else None
    if frame is None:
        raise SystemExit(f"Couldn't grab a snapshot from {name}:{no} over RTSP.")
    select_region(f"{name}:{no} live snapshot", cam_dir / "region.json", frame)


# --------------------------------------------------------------------------

def _common(p):
    p.add_argument("--camera", required=True,
                   help="Comma-separated <NVR name>:<camera number>, e.g. home-hik:15,shop-dahua:1 "
                        "(NVRs are listed in nvr-sdk/nvrs.env).")
    p.add_argument("--root", default=str(default_root()),
                   help="Where camera folders live (default: NVR_SCAN_ROOT in .env, else ./nvr_scans).")


def _detector_args(p):
    p.add_argument("--targets", default="animal",
                   help=f"Comma-separated Targets (groups: "
                        f"{', '.join(f'{k}={'+'.join(v)}' for k, v in osc.CLASS_GROUPS.items())}). Default: animal.")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--imgsz", type=int, default=None,
                   help="Default 640 for a camera with a Region, 1280 without.")
    p.add_argument("--gap", type=float, default=osc.DEFAULT_GAP_S,
                   help="Max seconds between Hits merged into one Sighting.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--model", default="yolo26x.pt",
                   help="Default yolo26x.pt (see DESIGN.md), run as a TensorRT engine when TensorRT is installed.")
    p.add_argument("--no-trt", action="store_true", help="Run the .pt with PyTorch instead of a TensorRT engine.")


def main(argv):
    parser = argparse.ArgumentParser(prog="object_scan.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("nvr", help="Object Scan over recorded footage pulled from NVRs.")
    _common(p)
    p.add_argument("--from", dest="start", required=True,
                   help="Start time: HH:MM:SS (today) or 'YYYY-mm-dd HH:MM:SS'.")
    p.add_argument("--direction", choices=["forward", "backward"], default="forward",
                   help="forward: from --from up to now (default). backward: from --from back into the past.")
    p.add_argument("--end", default="1h",
                   help="backward only: how far back to go from --from (default 1h; e.g. 90m, 6h).")
    p.add_argument("--chunk-min", type=int, default=5, help="Download chunk length in minutes (default 5).")
    p.add_argument("--connections", type=int, default=4,
                   help="Max concurrent downloads per NVR, shared fairly by its cameras "
                        "(default 4; some NVRs error under more).")
    p.add_argument("--prefetch", type=int, default=2, help="Chunks downloaded ahead per camera (default 2). Up to this + 1 chunks "
                        "of one camera download in parallel, still capped by --connections.")
    p.add_argument("--chunk-timeout", type=int, default=300, help="Seconds before a chunk download is abandoned.")
    p.add_argument("--full-download", action="store_true",
                   help="Dahua: download full footage instead of fetching keyframes only (the default for "
                        "Dahua; Hikvision always downloads full footage).")
    p.add_argument("--vram-limit", type=float, default=95,
                   help="Start another keyframe decoder (ffmpeg/NVDEC) only while GPU memory use is under this "
                        "percent and the detector is short of frames (default 95). One decoder always runs.")
    p.add_argument("--restart", action="store_true", help="Forget which stretches were already scanned.")
    _detector_args(p)

    p = sub.add_parser("live", help="Live Watch: live RTSP stream of each camera, one shared detector.")
    _common(p)
    _detector_args(p)

    p = sub.add_parser("select-region", help="Pick a camera's Region on a live snapshot.")
    _common(p)

    args = parser.parse_args(argv)
    if args.cmd == "nvr":
        if args.direction == "forward" and args.end != "1h":
            print("[object-scan nvr] --end only applies to --direction backward; ignored.")
        try:
            run_nvr(args)
        except KeyboardInterrupt:   # before scanning started (login, model load)
            print("\n[object-scan nvr] stopped (Ctrl+C).", flush=True)
            os._exit(130)
    elif args.cmd == "live":
        run_live(args)
    else:
        run_select_region(args)
