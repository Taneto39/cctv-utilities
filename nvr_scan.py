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
    sightings/
      sightings.csv        one row per Sighting, all runs, rebuilt as Hits arrive
      images/              best keyframe per Sighting (boxes + Region outline)
      frames/              every keyframe that had a Hit (images are drawn from these,
                           since the downloaded footage itself is deleted)
      hits_nvr.jsonl       Hits from recorded footage   } Sightings are built from both,
      hits_live.jsonl      Hits from Live Watch         } duplicates counted once
      .nvr_state.json      which stretches of recorded time are already scanned

Recorded footage is downloaded in chunks aligned to a fixed wall-clock grid
(default 5 min), scanned keyframe-by-keyframe, then deleted.
"""
import argparse
import csv
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from collections import Counter
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
        from ultralytics import YOLO  # deferred: slow import
        self.model = YOLO(args.model)
        self.class_targets = osc.resolve_targets(self.model.names, args.targets)
        self.class_ids = sorted(self.class_targets)
        self.conf = args.conf
        self.batch_size = args.batch_size
        self.stores = stores
        self.crops = {}
        self.imgsz = {}
        self.n_frames = 0
        self.n_hits = Counter()
        self.on_hits = None
        for cam_id, store in stores.items():
            self.imgsz[cam_id] = args.imgsz or (640 if store.region else 1280)

    def fingerprint(self, cam_id, model_path):
        st = self.stores[cam_id]
        return osc.fingerprint(self.class_targets, self.conf, self.imgsz[cam_id], model_path, st.region)

    def _crop(self, cam_id, frame):
        store = self.stores[cam_id]
        if not store.region:
            return None
        if cam_id not in self.crops:
            h, w = frame.shape[:2]
            self.crops[cam_id] = osc.region_crop_box(store.region["points"], w, h)
        return self.crops[cam_id]

    def _run_batch(self, batch):
        # one predict() per imgsz (cameras with / without a Region differ)
        by_size = {}
        for item in batch:
            by_size.setdefault(self.imgsz[item[1]], []).append(item)
        for imgsz, items in by_size.items():
            imgs, offs = [], []
            for _, cam_id, t, frame, src in items:
                crop = self._crop(cam_id, frame)
                if crop:
                    x0, y0, x1, y1 = crop
                    imgs.append(frame[y0:y1, x0:x1])
                    offs.append((x0, y0))
                else:
                    imgs.append(frame)
                    offs.append((0, 0))
            results = self.model.predict(imgs, imgsz=imgsz, conf=self.conf, classes=self.class_ids, verbose=False)
            for (_, cam_id, t, frame, src), (ox, oy), r in zip(items, offs, results):
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
        self.n_frames += len(batch)

    def run(self, q, n_producers, on_hits=None, idle=None):
        """Consumes q until every producer has sent "done". `idle` (if given)
        is called whenever the queue has been empty for a moment."""
        self.on_hits = on_hits
        remaining = n_producers
        batch = []
        while remaining > 0:
            try:
                item = q.get(timeout=0.5)
            except queue.Empty:
                if batch:
                    self._run_batch(batch)
                    batch = []
                if idle:
                    idle()
                continue
            kind = item[0]
            if kind == "frame":
                batch.append(item)
                if len(batch) >= self.batch_size:
                    self._run_batch(batch)
                    batch = []
                continue
            if batch:
                self._run_batch(batch)
                batch = []
            if kind == "mark":
                item[2]()
            elif kind == "done":
                remaining -= 1


# --------------------------------------------------------------------------
# Recorded footage (Object Scan, NVR source)
# --------------------------------------------------------------------------

class NvrCameraJob:
    """Downloads one camera's chunks (a little ahead) and decodes their
    keyframes into the shared detector queue."""

    def __init__(self, cam, store, chunks, q, args, stats, pbar_update):
        self.cam, self.store, self.chunks, self.q = cam, store, chunks, q
        self.args, self.stats, self.pbar_update = args, stats, pbar_update
        self.ready = queue.Queue(maxsize=args.prefetch)
        self.tmp = store.dir / ".chunks"
        self.tmp.mkdir(exist_ok=True)

    def start(self):
        threading.Thread(target=self._download_all, daemon=True).start()
        threading.Thread(target=self._decode_all, daemon=True).start()

    def _download_all(self):
        ext = ".dav" if self.cam.cfg["vendor"] == "dahua" else ".mp4"
        for cs, ce in self.chunks:
            # either direction: a chunk ending near "now" may not be fully recorded yet
            wait = _ts(ce) + SETTLE_S - _ts(datetime.now())
            if wait > 0:
                print(f"[{self.store.cam_id}] waiting {wait:.0f}s for {ce:%H:%M:%S} to finish recording on the NVR")
                time.sleep(wait)
            dst = self.tmp / f"{cs:%Y%m%d-%H%M%S}{ext}"
            result = False
            for attempt in range(DOWNLOAD_RETRIES + 1):
                try:
                    result = self.cam.download(cs, ce, str(dst), timeout=self.args.chunk_timeout)
                except Exception as e:  # SDK hiccup: retry like a failed download
                    print(f"[{self.store.cam_id}] download {cs:%H:%M:%S} error: {e}")
                    result = False
                if result is not False:
                    break
                if attempt < DOWNLOAD_RETRIES:
                    time.sleep(5 * 3 ** attempt)
            self.ready.put((cs, ce, dst if result else None, result))
        self.ready.put(None)

    def _decode_all(self):
        cam_id = self.store.cam_id
        try:
            while True:
                item = self.ready.get()
                if item is None:
                    break
                cs, ce, path, result = item
                ok = True
                if path is not None:
                    try:
                        for offset, frame in osc.iter_keyframes(path):
                            self.q.put(("frame", cam_id, cs + timedelta(seconds=offset), frame, "nvr"))
                    except Exception as e:
                        print(f"[{cam_id}] decode {cs:%H:%M:%S} failed: {e}")
                        ok = False
                    finally:
                        try:
                            os.remove(path)
                        except OSError:
                            pass
                self.q.put(("mark", cam_id, lambda cs=cs, ce=ce, r=result, ok=ok: self._chunk_done(cs, ce, r, ok)))
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
        self.pbar_update((ce - cs).total_seconds())
        self.store.write_outputs(self.args.gap)


def run_nvr(args):
    from tqdm import tqdm
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
            cams[cam_id] = nvr.Camera(name, no, max_connections=args.connections, nvrs=nvrs)
        except (RuntimeError, SystemExit) as e:
            print(f"[{cam_id}] skipped: {e}")
            continue
        stores[cam_id] = CameraStore(root, cam_id)
    if not cams:
        raise SystemExit("No camera could be opened.")

    det = Detector(args, stores)
    q = queue.Queue(maxsize=args.batch_size * 8)
    stats = {c: {"ok": 0, "no_footage": 0, "failed": [], "skipped": 0} for c in cams}
    jobs, total_s = [], 0.0
    for cam_id, cam in cams.items():
        store = stores[cam_id]
        store.load_coverage(det.fingerprint(cam_id, args.model), args.restart)
        todo = [(cs, ce) for cs, ce in chunks_all if not store.is_covered(cs, ce)]
        stats[cam_id]["skipped"] = len(chunks_all) - len(todo)
        total_s += sum((ce - cs).total_seconds() for cs, ce in todo)
        if todo:
            jobs.append((cam_id, NvrCameraJob(cam, store, todo, q, args, stats, None)))
    print(f"[object-scan nvr] {len(cams)} camera(s), {args.direction} {lo:%Y-%m-%d %H:%M:%S} -> "
          f"{hi:%Y-%m-%d %H:%M:%S}, {len(chunks_all)} chunk(s) of {args.chunk_min} min each per camera"
          + "".join(f"\n  {c}: {stats[c]['skipped']} chunk(s) already scanned" for c in cams if stats[c]["skipped"]))

    t0 = time.time()
    try:
        with tqdm(total=round(total_s), unit="s", desc="footage", dynamic_ncols=True,
                  bar_format="{l_bar}{bar}| {n:.0f}/{total:.0f}s footage [{elapsed}<{remaining}]") as pbar:
            for _, job in jobs:
                job.pbar_update = pbar.update
                job.start()
            det.run(q, len(jobs))
    finally:
        for cam in cams.values():
            cam.close()
    elapsed = time.time() - t0
    if det.n_frames:
        print(f"[object-scan nvr] {det.n_frames} keyframes in {elapsed:.0f}s "
              f"({total_s / max(elapsed, 1e-6):.0f}x real time over all cameras)")
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
        cams[cam_id] = _LiveCam(nvrs[name], no)
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


class _LiveCam:
    """RTSP URL only -- Live Watch doesn't need an SDK login."""

    def __init__(self, cfg, no):
        self.cfg, self.camera_no = cfg, no

    def rtsp_url(self):
        import nvr
        return nvr.Camera.rtsp_url(self)


# --------------------------------------------------------------------------
# select-region on a live snapshot
# --------------------------------------------------------------------------

def run_select_region(args):
    nvr = import_nvr()
    nvrs = nvr.load_nvrs()
    (name, no), = nvr.parse_cameras(args.camera)
    if name not in nvrs:
        raise SystemExit(f"No NVR named '{name}' in nvrs.env")
    cam = _LiveCam(nvrs[name], no)
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
    p.add_argument("--model", default="yolo26x.pt", help="Default yolo26x.pt (see DESIGN.md).")


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
                   help="Max concurrent downloads per NVR (default 4; some NVRs error under more).")
    p.add_argument("--prefetch", type=int, default=2, help="Chunks downloaded ahead per camera (default 2).")
    p.add_argument("--chunk-timeout", type=int, default=300, help="Seconds before a chunk download is abandoned.")
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
        run_nvr(args)
    elif args.cmd == "live":
        run_live(args)
    else:
        run_select_region(args)
