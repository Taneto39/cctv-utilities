# cctv-utilities

Tools for working with CCTV/DVR/NVR footage:

- **Motion scan** -- pull motion-event clips out of long DVR exports
  (Hikvision-style, chained into many small `.mp4` segments per day).
- **Clip filter** -- run YOLO over those clips and keep only the ones that
  contain an object of interest (person/dog/cat by default).
- **Object scan** -- run YOLO directly over raw footage (keyframes only) and
  get wall-clock time ranges ("Sightings") when a target appears, from files
  or straight from a Hikvision/Dahua NVR.
- **Censor** -- privacy-blur/black-out a region in a folder of already-cut
  clips.

## Layout

```
motion_scan.py           motion detection: pick a region once, then scan
detect_objects.py        second-pass YOLO filter over motion_scan.py's clips
object_scan.py           YOLO over raw footage (keyframes) -> Sightings
nvr_scan.py              NVR download / live-watch source for object_scan.py
sighting_wall.py         2x2 viewer for Sightings as they arrive
select_region.py         region picker UI + region.json format
censor.py                blur/black-out/crop a region in a flat folder of clips
rename_cctv_download.py  rename numeric NVR exports to sortable timestamps
bench_decord_gpu.py      one-off decode benchmark (decord/NVDEC vs OpenCV)
gcloud_billing_function/ Cloud Function that disables billing at a budget cap
```

Each **camera folder** is self-contained:

```
<folder>/
  data/          source footage
  region.json    detection zone (created by select-region)
  output/        event clips written by scan
```

No test suite or build step -- run the scripts directly with `python`.
`ffmpeg`/`ffprobe` must be on `PATH`.

## Install

```bash
pip install -r requirements.txt
```

`motion_scan.py` alone only needs `opencv-python numpy tqdm`; see the
comments in [requirements.txt](requirements.txt) for the optional CUDA torch
install.

## motion_scan.py

```bash
# 1. Pick the detection zone once per camera folder. Click to add points,
#    right-click to undo, N to close the current shape and start another
#    (shapes are merged into one region), Enter/S to save, Esc/Q to cancel.
python motion_scan.py select-region --folder /path/to/cam1

# 2. Scan every file in <folder>/data, write clips to <folder>/output
python motion_scan.py scan --folder /path/to/cam1 --workers 6
```

`--input` / `--region` / `--output` override the `--folder` defaults (e.g.
to share one `region.json` across cameras with the same view).

| flag | default | what it does |
|---|---|---|
| `--threshold` | 0.15 | fraction of the region that must be foreground to count as motion |
| `--min-event-len` | 1.0s | how long motion must last before a clip starts |
| `--pre` / `--post` | 2.0s / 2.0s | padding kept before/after each event |
| `--workers N` | `cpu_count-1` | parallel processes. Throughput typically plateaus around 6-10; an event straddling two workers' chunks may be split into two clips |
| `--gpu` | off | PyTorch CUDA motion detector instead of CPU/MOG2. Slower in testing **and misses low-contrast night/IR motion** -- don't use it on night footage |
| `--nvdec` / `--nvdec-workers N` | off / 0 | GPU decode in all / N workers. Measured slower than CPU decode; kept for experiments |
| `--dynamic` | off | hand out work from a shared queue instead of a fixed upfront split (only helps when mixing decode backends) |
| `--restart` | off | ignore saved progress and start over |
| `--manifest-only` | off | log events to `<output>/events_manifest.jsonl` instead of writing clips (see below) |
| `--note TEXT` | - | label stored in this run's log entry |

**Resume:** an interrupted run (at `--workers 2` or more) resumes at segment
granularity as long as the inputs and settings are unchanged (`--workers`
itself may change).

**Run log:** every `scan` appends `start` / `checkpoint` / `end` records to
`<output>/scan_runs.jsonl`, including wall-clock `fps`. Use that for
throughput numbers rather than the progress bars, which are smoothed.

### Scanning on a cloud VM: `--manifest-only` + `extract`

Uploading footage to a cloud VM is usually free; downloading results is
not. `--manifest-only` writes only a small JSONL manifest (source file names
+ frame ranges), which you download and then cut into real clips locally
against your own copy of the footage:

```bash
# on the VM
python motion_scan.py scan --folder /path/to/cam1 --workers 8 --manifest-only

# locally, after downloading events_manifest.jsonl
python motion_scan.py extract --folder /path/to/cam1 \
    --manifest /path/to/cam1/output/events_manifest.jsonl --workers 8
```

File names in the manifest are bare (no directory), so a manifest produced
on Linux works against a Windows `--input` and vice versa. Deleting files on
the VM doesn't reduce cost (disks bill by provisioned size) -- delete the VM
and its disk when done.

## detect_objects.py

Second pass over `motion_scan.py`'s clips: reports which clips contain the
requested classes and copies matches to `<folder>/detected/` as soon as each
one is found.

```bash
python detect_objects.py --folder /path/to/cam1                 # copy matches
python detect_objects.py --folder /path/to/cam1 --no-copy       # report only
python detect_objects.py --folder /path/to/cam1 --move          # move instead of copy
python detect_objects.py --folder /path/to/cam1 --classes dog,cat,bird
python detect_objects.py --folder /path/to/cam1 --workers 3
```

Classes must exist in the model's vocabulary (COCO for the default
`yolo11x.pt`). `--batch-size` (default 16) trades VRAM for GPU utilization;
`--workers` is automatically reduced if the requested workers x batch size
wouldn't fit in free VRAM.

## object_scan.py

Finds a Target directly in raw footage, looking only at keyframes, and
merges detections into **Sightings** (wall-clock time ranges). Where
`detect_objects.py` answers "does this short clip contain a cat?", this
answers "when did the cat show up today?".

```bash
# camera folder: reads <folder>/data, uses <folder>/region.json if present,
# writes <folder>/sightings/
python object_scan.py --folder /path/to/cam1

# any folder (searched recursively); output goes to <input>/sightings/
python object_scan.py --input /path/to/recordings --targets animal,person,car

# regroup existing hits with a different gap -- no rescan
python object_scan.py --input /path/to/recordings --gap 30
```

Output (`sightings/`): `sightings.csv` (one row per Sighting), `images/`
(best keyframe per Sighting, annotated), `hits.jsonl` (raw detections).

Notes:

- Re-runs only scan new/unfinished files; changing targets, confidence,
  image size, model or region starts over (`--restart` forces it).
- With a region, a crop around it is scanned at `--imgsz 640`; without one,
  the full frame is scanned at 1280 so small objects (~20px) aren't lost.
  The region ranks Sightings by distance, it never filters them out.
- Objects need to be roughly 15-20px or larger to be detected.
- Wall-clock time comes from file names (`2026-08-24_044612.mp4` or raw NVR
  names containing start/end timestamps).
- Decoding uses NVDEC (`-hwaccel cuda`), which is faster than CPU for
  keyframe-only decode.

**Targets:** any COCO class name (`person`, `cat`, `dog`, `car`, `truck`,
`bicycle`, `bird`, ...), or a group -- `animal` = `cat` + `dog` (default),
since small cats are often labelled `dog`. Groups live in `CLASS_GROUPS` in
`object_scan.py`. Quote names containing spaces: `--targets "person,cell phone"`.
The default model is `yolo26x.pt`; change it with `--model`.

### Directly from an NVR (Hikvision / Dahua), live watch, Sighting Wall

Downloads footage from the NVR in chunks (default 5 minutes) through the
vendor SDKs, scans each chunk and deletes it, keeping only results. Windows
only (the SDKs are Windows DLLs). The vendor SDKs are used rather than
ISAPI.

Setup: put the SDK wrappers in a folder of your choice and point
`NVR_SDK_DIR` at it in `.env` (see [.env.example](.env.example)); NVR
addresses and credentials go in that folder's `nvrs.env`.

Cameras are `<nvr name>:<channel>`, comma-separated. All cameras share one
model instance.

```bash
# from a past time forward until now, then exit
python object_scan.py nvr --camera hik1:3,dahua1:1 --from 09:00:00
python object_scan.py nvr --camera hik1:3 --from "2026-10-07 22:00:00"

# newest first, back over --end (default 1h)
python object_scan.py nvr --camera hik1:3 --from 10:00:00 --direction backward --end 3h

# live, Ctrl+C to stop
python object_scan.py live --camera hik1:3,dahua1:1

# pick a region from the live image
python object_scan.py select-region --camera hik1:3

# 2x2 wall of Sightings as they arrive (q/Esc quit, f fullscreen)
python sighting_wall.py
```

Run `nvr` and `live` at the same time for "catch up, then keep watching".
Results accumulate per camera under `<root>/<nvr>_ch<N>/sightings/`
(`NVR_SCAN_ROOT` in `.env` or `--root`, default `./nvr_scans`).
Already-scanned time is skipped on later runs; failed chunks are retried on
the next run. Speed is network-bound -- NVRs send every frame, so
keyframe-only download isn't possible. Live timestamps lag the burned-in
clock by a few seconds.

## censor.py / select_region.py

Separate from the pipeline above: hides a region in every clip of a flat
folder of already-cut clips.

```bash
# 1. pick the region once (saved as <folder>/region.json); points can be
#    dragged after placing
python select_region.py --folder /path/to/clips

# 2. censor every clip -> <folder>/censored/ (existing outputs skipped
#    unless --force)
python censor.py --folder /path/to/clips
```

| `--effect` | what it does |
|---|---|
| `black` (default) | solid black fill, hard edge -- cheapest |
| `blur` | gaussian blur (`--blur-sigma`, default 55); add `--gpu` for PyTorch CUDA (much faster than CPU blur) |
| `crop` | crop the frame to the region's bounding box. Here the region means *keep*, not *hide*, so pick a separate region file for it |

```bash
python censor.py --folder /path/to/clips --effect blur --gpu
python select_region.py --folder /path/to/clips --region /path/to/clips/crop_region.json
python censor.py --folder /path/to/clips --effect crop --region /path/to/clips/crop_region.json
```

`--frame-source <clip>` picks which clip's first frame to preview. One bad
clip doesn't stop the batch.

## Other scripts

- `rename_cctv_download.py [folder]` -- renames Hikvision NVR exports from
  numeric names to sortable timestamps using the exported file-list `.txt`.
- `bench_decord_gpu.py` -- one-off decode benchmark, not part of the
  workflow.

## Notes on Hikvision exports

Some Hikvision NVR `.mp4` exports are really an `IMKH` header followed by
MPEG-PS, with no seek index. The scripts handle this fine, but GUI players
may seek or fast-forward poorly. Remux before manual review:

```bash
ffmpeg -i in.mp4 -c copy -movflags +faststart out.mp4
```

Footage folders (`data/`, `output/`, video files) are gitignored.
