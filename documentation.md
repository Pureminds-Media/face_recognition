# System Documentation

A comprehensive reference for the Face Recognition Attendance System — architecture, data flow, database schema, configuration, and internal behaviour.

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Process Model](#2-process-model)
3. [Detection & Recognition Pipeline](#3-detection--recognition-pipeline)
4. [Camera & Grid System](#4-camera--grid-system)
5. [Visit Lifecycle](#5-visit-lifecycle)
6. [Auto-Capture (Unknown Persons)](#6-auto-capture-unknown-persons)
7. [Footage Recording](#7-footage-recording)
8. [Action Detection](#8-action-detection)
9. [Head Detection](#9-head-detection)
10. [Hardware Decode (NVDEC)](#10-hardware-decode-nvdec)
11. [Database](#11-database)
12. [Configuration Reference](#12-configuration-reference)
13. [File Layout](#13-file-layout)
14. [Web UI Pages](#14-web-ui-pages)
15. [Gate Events & Daily Reports](#15-gate-events--daily-reports)
16. [Analytics & Reports](#16-analytics--reports)
17. [Camera Tracker (Line-Crossing)](#17-camera-tracker-line-crossing)
18. [Manual Attendance](#18-manual-attendance)
19. [UI Login & User Management](#19-ui-login--user-management)
20. [Manual Detect](#20-manual-detect)

---

## 1. High-Level Architecture

```
Browser / Remote Client
        │  HTTP / SSE / MJPEG
        ▼
  Flask Server (app.py)          ← main OS process
        │  multiprocessing.Pipe
        ▼
  Engine Subprocess (engine_runner.py + face_engine.py)
        │                   │
  Camera capture          InsightFace pool
  (cv2 / PyAV NVDEC)      (ONNX Runtime GPU)
        │                   │
  OpenCV trackers    Head detector (YOLOv8n)
                     Action detector (CLIP)
```

The Flask server and the face-recognition engine run in **separate OS processes** connected by a `multiprocessing.Pipe`. This isolates GPU crashes (SIGSEGV from ONNX/NVDEC) from the web server — if the engine dies, the watchdog in `app.py` automatically respawns it.

---

## 2. Process Model

### Flask server (`app.py`)

- Serves all HTTP endpoints and the MJPEG feed.
- Owns the SQLite/PostgreSQL database connection.
- Holds an `EngineClient` that sends commands over the Pipe and reads state from a shared `Manager` dict.
- Runs a `_watchdog_loop` thread: polls `proc.exitcode` every second; if the subprocess dies, calls `_start_engine()` to respawn.
- Runs a `_visit_manager_loop` thread: reads attendance events emitted by the engine and opens/updates/closes visit rows in the database.
- Emits Server-Sent Events to connected browsers via `/api/attendance/stream`.

### Engine subprocess (`engine_runner.py`)

- Imports and instantiates `FaceEngine`.
- Runs `_dispatch()` in a loop, reading commands from the Pipe:
  - `start` — launches `engine.start()` in a background thread (so camera opens don't block the command loop).
  - `stop` — calls `engine.stop()`.
  - `set_viewer` — updates which camera is shown in the MJPEG feed.
  - `reload_faces` — triggers a synchronous embedding rebuild.
  - `get_jpeg`, `get_tracks`, `get_status`, etc. — read-only queries.
- Publishes live state (jpeg bytes, track list, running flag, fps) to the shared `Manager` dict so the Flask process can read it without round-trip latency.

### Exception logging

Both `app.py` and `engine_runner.py` install a `threading.excepthook` and `sys.excepthook` that route uncaught exceptions through the standard `logging` module (and therefore into `logs/app.log`) instead of letting them print to stderr only. Without this, an exception in a background thread (camera worker, footage writer thread, etc.) would be visible only in whatever terminal launched the process — invisible if that terminal isn't being watched or has since closed. Note this does **not** catch process kills from outside Python (e.g. the OOM killer's `SIGKILL`) — those still leave no trace beyond a gap in the logs and a fresh "Database initialised" line when the watchdog or manual restart brings the process back. Check `dmesg`/`journalctl -k` for OOM kills if a silent gap appears with no corresponding exception logged.

### Inter-process communication

| Channel | Direction | Used for |
|---------|-----------|----------|
| `multiprocessing.Pipe` | bidirectional | Commands (Flask → engine) and responses |
| `multiprocessing.Manager` dict | engine writes, Flask reads | Live JPEG, tracks, status, fps |

### `EngineClient` concurrency (`engine_client.py`)

Multiple Flask request threads can call into the engine concurrently, all serialized through one `multiprocessing.Pipe`, so `EngineClient._call()` has to guarantee that a thread's `send()` is matched with *its own* `recv()` and not another thread's. Two real bugs were found and fixed here:

- **Lock-swap race during watchdog respawn.** When the watchdog (`_watchdog_loop` in `app.py`) detects the engine subprocess has died and respawns it, the `EngineClient` needs a new `Pipe` connection. Previously the respawn path replaced both `self._conn` and `self._lock` with new objects. A request thread already blocked inside `with self._lock:` on the *old* lock object had no mutual exclusion against a new request that read `self._lock` *after* the swap and acquired the *new* lock — so two threads could `send()`/`recv()` on the pipe at the same time and each could receive the other's response (e.g. a `manual_detect` call receiving a `stop_footage` tuple). Fixed by never replacing `self._lock` — it's created once in `__init__` and every caller, past and future, serializes on the exact same lock object. The respawn path now goes through `EngineClient._swap_connection(new_conn)`, which takes that same lock before touching `self._conn`, so a swap can't interleave with an in-flight call.
- **Timeout-orphaned response desync.** If `_call()` times out waiting for `self._conn.poll(timeout)`, the engine is typically still processing the original command and will eventually send a response anyway. Previously, timing out just raised immediately without reading that eventual response — it would sit in the pipe and get handed to whichever unrelated `_call()` happened to `recv()` next (e.g. `manual_detect` times out, then a later `manual_detect_region` call's `recv()` pops `manual_detect`'s stale response instead of its own). Fixed by draining the late response (`self._conn.poll(30)` + `recv()`, discarded) before raising the timeout error, while still holding the lock — so the pipe is back in sync before any other caller can send its next command.

---

## 3. Detection & Recognition Pipeline

### Face embeddings (`faces/` directory)

- Each enrolled person has a subdirectory: `faces/<name>/`.
- Supported image formats: `.jpg`, `.jpeg`, `.png`, `.webp`.
- On startup (and after `reload_faces`), every image is run through InsightFace to extract a 512-d ArcFace embedding. Results are cached in `faces/<name>/.arcface.npz` to avoid re-encoding on every restart.
- The final template for each person is the **mean** of all their embeddings, L2-normalised.

### InsightFace pool

- `INFERENCE_POOL_SIZE` (default **6**) independent `FaceAnalysis("buffalo_l")` instances are pre-loaded at engine start.
- Each instance has its own ONNX Runtime CUDA session / stream, so concurrent calls from different camera detect threads genuinely overlap on the GPU rather than serialising.
- Threads acquire an instance from the pool via a `threading.Semaphore` and return it when done.
- Memory cost: ~300–400 MB VRAM per instance. At `INFERENCE_POOL_SIZE=6` on an 8 GB GPU (RTX 4060 laptop), ~2.4 GB is reserved for inference, leaving headroom for NVDEC and OS.

### Detection cadence

- Each camera runs a **detect thread** that fires every `detect_every` seconds (default **1.0 s**, but the value passed from `app.py` is **5 s** for the shared pool).
- Between detections, OpenCV **CSRT trackers** (one per active track) keep the bounding boxes alive at full frame rate.
- Head detection (YOLOv8n) supplements face detection to sustain tracking when the face is not visible.

### Motion gating

Before calling InsightFace, each detect thread runs a fast CPU motion check: the current frame is downscaled to 160×90 grayscale and compared against the previous processed frame using `cv2.absdiff`. GPU inference is skipped if the number of changed pixels (diff > 15) is below `motion_thresh` (default **500**). Controlled by `motion_gate` (default **true**). Reduces GPU utilisation by 60–80% on static scenes.

### Per-camera motion gate override

The global `motion_gate` / `motion_thresh` settings can be overridden per camera. Each camera entry in `ip_cameras.json` may carry `motion_gate_enabled` (bool) and `motion_threshold` (int) fields, set from **Settings → Camera** (a checkbox + a threshold number input per camera row). Whenever these are saved, `app.py`'s `_push_motion_gate_overrides()` builds a `{camera_source: {"enabled": bool, "threshold": int}}` dict from every camera that has explicitly set one or both fields, and pushes it to the engine via `FaceEngine.set_config({"motion_gate_overrides": {...}})`. `FaceEngine` stores this as `self.motion_gate_overrides` and, in the grid detect loop, looks up the current camera's source in that dict before falling back to the engine-wide `motion_gate`/`motion_thresh` values. Cameras with no override entry behave exactly as before (governed by the global setting).

### High-priority cameras

Cameras listed in `high_priority_sources` (a set of RTSP URLs) bypass the motion gate entirely and always run detection at full resolution (`detect_scale = 1.0`), regardless of the engine-wide `detect_scale` setting. The gate cameras (arrival + exit) are registered here at startup from `reports_config.json`.

### Adaptive detect_every backoff

If a camera's worker queue depth exceeds 3 frames, the effective `detect_every` for that camera is multiplied by 1.5 (capped at `detect_every × 4`). This prevents GPU queue pile-up under load.

### Phase-offset stagger

Camera detect threads are staggered by `(i % pool_size) * (detect_period / pool_size)` to spread GPU inference evenly across the InsightFace pool rather than firing all cameras simultaneously at t=0.

### Idle render FPS

The grid render thread tracks when the last MJPEG frame was requested (`_last_viewer_req_t`). If no viewer has polled for > 5 seconds, the render rate drops to 2 FPS. It snaps back to `out_fps` the moment a new request arrives. Configured via `ping_viewer()` called from the `/video` routes.

### Recognition matching

- Each detected face embedding is compared against all known person templates using **cosine distance**.
- Match threshold: `0.4` (default in `FaceEngine.__init__`; `app.py` passes `0.6`).
  - Distance < threshold → recognised as that person.
  - Distance ≥ threshold → labelled "unknown".
- Confidence stored in the visit row is the minimum cosine distance seen across all detections in that visit (lower = more confident).

---

## 4. Camera & Grid System

### Single-camera mode

- `cam_index` is an integer (webcam index) or URL string (RTSP/HTTP).
- The engine runs one capture loop and one detect loop.
- The viewer always shows this camera.

### Grid mode

- Activated when `grid_config.json` exists with a valid layout.
- Supported layouts: 2×2, 3×3, 4×4 (up to 16 cameras).
- Each camera slot gets its own **capture thread** (`_grid_capture_loop`) and **detect thread** (`_grid_detect_loop`).
- A separate **render thread** (`_grid_render_loop`) composites tiles into the MJPEG frame.
- The analysis pool covers **every configured camera** simultaneously — switching the viewer mode never starts or stops detection.
- Viewer controls:
  - `viewer_mode = "single"` — shows one camera tile full-screen.
  - `viewer_mode = "grid"` — shows the composite.
  - `viewer_grid_offset` — pages through more cameras than the layout has slots.

### Camera reconnection

- If a grid camera's `cap.read()` fails, a background thread (`_reconnect`) waits 3 seconds then reopens the capture with `open_capture()`.
- Uses `hw_capture.open_capture()` which tries NVDEC first, falls back to `cv2.VideoCapture` on failure.

### Ghost-box / track staleness (grid detect loop)

A track (bounding box) can be kept alive by the head detector reconfirming it even without a fresh face detection — this bridges brief occlusions (person turns away, walks behind an obstacle) without a jumpy cut in tracking or footage. Two windows control how long a track survives without reconfirmation, in `_grid_detect_loop` (`face_engine.py`):

- `head_reconfirm_secs = max(3.0, detect_every * 5.0)` — how long the head detector alone can sustain a track.
- `known_reconfirm_secs` — how long a known person's track can go without *any* reconfirmation (face or head) before being dropped. Previously computed adaptively from GPU contention (`detect_every * n_workers`, capped at 30 s) to avoid killing real tracks when detection cadence stretched out under load with many cameras sharing the GPU. Currently a flat **5.0 s** — traded some robustness against detector slowdowns for tighter bounding on how long a track (and its associated visit/footage recording) lingers after a person actually leaves frame. If visits start fragmenting unexpectedly under heavy multi-camera load (a real visit split into several short ones), this is the first place to look — raise it back toward the old adaptive formula.

Because footage recording and visit closure are driven by whether *any* track exists for a person (not by a dedicated "person has left" signal), this timeout directly controls how long recording continues after someone is actually gone.

### RTSP connection pool pinning

`RtspConnectionPool` supports pinning sources (`pin()`, `unpin()`, `set_pinned()`) so they are immune to LRU eviction under `MAX_CONCURRENT_RTSP` pressure. Cameras assigned to the [Camera Tracker](#17-camera-tracker-line-crossing) are automatically pinned and added to `high_priority_sources` (full-res, no motion gate) so their connections are never dropped.

### IP camera groups

- Configured via the UI under Settings → IP Cameras.
- Stored as groups (e.g. "Floor 1") containing cameras with a name and RTSP URL or channel number.
- `POST /api/ip_cameras/groups/<id>/cameras` accepts `{name, channel}` (builds URL from group `base_url`) or `{name, url}` (explicit RTSP URL).
- Each group has a `branch` field (e.g. `"Riyadh"` or `"Egypt"`). Visits opened by cameras in that group are automatically tagged with that branch. Defaults to `"Riyadh"` if not set.

### Grid config file (`grid_config.json`)

```json
{
  "layout": [2, 2],
  "slots": {
    "0": {"source": "rtsp://...", "name": "Entrance"},
    "1": {"source": "rtsp://...", "name": "Office"}
  }
}
```

---

## 5. Visit Lifecycle

A **visit** represents a continuous presence of one person at one camera location.

```
Person detected
      │
      ▼
open_visit()  ─── creates visits row (ended=0, first_seen=now, last_seen=now)
      │
      ▼
update_visit_seen()  ─── bumps last_seen each detection frame
      │
 (person disappears)
      │
      ▼
close_visit()  ─── sets ended=1, final last_seen
```

### Flip-flop prevention (`VISIT_TRANSITION_SECS`)

- When a person is seen on camera B while they have an active visit on camera A, the system waits `VISIT_TRANSITION_SECS` (default **30 s**) before closing the A visit and opening a new one on B.
- This prevents rapid visit churn in camera overlap zones.

### Timeout (`VISIT_TIMEOUT_MINUTES`)

- A janitor thread runs every 30 seconds and calls `close_stale_visits(timeout_minutes)`.
- Any open visit whose `last_seen` is older than `VISIT_TIMEOUT_MINUTES` (default **10 min**) is automatically closed.

### Visit fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | int | Auto-increment primary key |
| `person_name` | text | Name as stored in `faces/` directory |
| `location_id` | int FK | References `locations.id` |
| `first_seen` | ISO timestamp (UTC) | When the visit opened |
| `last_seen` | ISO timestamp (UTC) | Last detection frame |
| `ended` | bool/int | 0/false = open, 1/true = closed |
| `confidence` | float | Best (lowest) cosine distance seen |
| `session_id` | UUID FK | References `sessions.id` |
| `screenshot` | text | Filename of face crop image |
| `footage` | text | Filename of MP4 footage clip |
| `visible_duration` | float | Actual seconds on camera (footage writer clock) |
| `activity` | text | Most frequent CLIP action label during visit |
| `branch` | text | Branch this visit belongs to (e.g. `"Riyadh"`, `"Egypt"`). Auto-assigned from the camera's IP group. |

### `duration_secs` vs `visible_duration`

- `duration_secs` (serialised in API responses) = `last_seen − first_seen` — wall-clock elapsed time from first to last detection. Can be long if the person was mostly off-camera but briefly detected again.
- `visible_duration` = actual seconds the footage writer was running (i.e., seconds the person was continuously tracked). Used by the Top 10 Longest Working analytics endpoint.

---

## 6. Auto-Capture (Unknown Persons)

Controlled by `AUTO_CAPTURE_ENABLED` env var (default **false**).

### Enrolment flow

1. An unrecognised face is tracked. The engine accumulates frames and keeps the highest-quality crop (largest bounding box area).
2. `_unknown_capture_min_seconds` (currently **0** — capture as soon as a track exists) gates how long a track must be held before the best crop is considered.
3. **TEMPORARY:** the InsightFace re-detection quality gate (`det_score ≥ 0.50`, min face size, embedding-based dedup) is disabled — crops are saved directly from the tracker bbox once they are ≥ 64×64 px. This is a testing change to speed up capture. Production defaults: `_unknown_capture_min_seconds = 1.5`, quality gate enabled.
4. On pass: image saved to `faces/unknown_N/1.jpg`; track promoted to `unknown_N`.

### Bootstrap accumulation

- For the first **10 images** after enrolment, additional samples are saved with only a **5-second cooldown** per camera (instead of the steady-state 10 minutes).
- This quickly builds a diverse averaged template, fixing the "single-frame template never matches re-appearances" failure mode.

### Demotion

- Tracks that lose recognition for **5 consecutive misses** (`track_max_misses`, up from 3) are dropped from the active track list.

### Limits

- Maximum **30 images** per person.
- Maximum **150** auto-captured `unknown_N` folders total.
- Maximum **10** simultaneous unknown tracks per camera (`max_unknown_tracks`, up from 3) — a testing change; see auto-capture note above.

---

## 7. Footage Recording

- Every open visit with a known person (not raw "unknown") starts a video writer for the visit's camera.
- **Encoder**: `_FFmpegWriter` (`face_engine.py`) pipes raw BGR24 frames to a system `ffmpeg` subprocess encoding H.264 (`libx264`), producing browser-playable `.mp4` output. This replaced `cv2.VideoWriter` with the `mp4v` (MPEG-4 Part 2) fourcc, which OpenCV's bundled FFmpeg build could produce but browsers cannot play. OpenCV's bundled FFmpeg only exposes `h264_v4l2m2m` (a hardware encoder requiring a V4L2 device, unavailable on typical x86/NVIDIA hosts) for H.264, so `cv2.VideoWriter` itself cannot produce H.264 here — hence the subprocess pipe to the system `ffmpeg` binary instead. If the `ffmpeg` binary can't be launched, falls back to the old `cv2.VideoWriter`/`mp4v` path (not browser-playable, but still a valid file).
- **Encoder tuning**: `-preset ultrafast -tune zerolatency -bf 0 -g <2×fps> -threads 2`. Chosen to minimize per-process memory and CPU — with many visits recording concurrently (one `ffmpeg` process each), default `libx264` settings (B-frames, larger lookahead buffers, one thread per core) scale badly: unconstrained, each instance used 600 MB–1.9 GB RAM under real camera load, enough to trigger the Linux OOM killer with ~20 concurrent recordings. The tuned flags bring this down to roughly 80 MB per instance.
- **Resolution cap** (`FOOTAGE_MAX_HEIGHT`, default `720`): frames are downscaled (aspect-ratio preserved, dimensions rounded to even for `yuv420p`) before the writer is opened. Native camera feeds can be up to 4K; software-encoding many concurrent 4K streams saturates CPU (each ~600–900% of one core observed at native res). A 4K feed downscaled to 720p is roughly a 9× reduction in encoded pixel count.
- Frames come from a **ring buffer** (`FOOTAGE_RING_SECS = 1.0 s`) so the clip starts slightly before the visit opened.
- Footage is written to `FOOTAGE_DIR` (required env var), normally a NAS mount. **Storage guard** (`_footage_storage_ok()` in `app.py`): before opening a writer, checks that `FOOTAGE_DIR` resolves (walking up parent directories) to an actual mounted filesystem (`os.path.ismount()`) with at least `FOOTAGE_MIN_FREE_BYTES` (default 1 GiB) free. If the NAS mount is down or full, recording is skipped entirely for that visit (logged as a warning) rather than silently writing to local disk underneath the unmounted mount point.
- On `close_visit()`, the writer is flushed and released; `visible_duration` is written to the DB.
- Footage files are served at `/footage/<filename>` (key-authenticated).
- Footage clips are **always annotated** (bounding boxes + labels) regardless of `LIVE_ANNOTATIONS_ENABLED`.
- **Retention**: `cleanup_footage.sh` (cron, daily) deletes footage files older than `FOOTAGE_RETENTION_DAYS` (default 7) from `FOOTAGE_DIR`. It refuses to run if the footage directory's parent isn't an actual mount (same guard rationale as above — never wants to prune local-fallback files thinking they're NAS footage). Cron output must redirect to a path the running user can write (`/var/log/` typically requires root) or the job fails silently every run with no error visible anywhere.

---

## 8. Action Detection

Controlled by `ACTION_DETECTION_ENABLED` env var (default **false**).

- Model: CLIP ViT-B/32 (ONNX Runtime GPU). Model files in `models/clip-vit-base-patch32-onnx/`.
- Zero-shot classification: the frame crop is scored against a fixed set of text labels (e.g. "Using phone", "Typing", "Idle").
- Runs in a **dedicated async thread** (`_activity_loop`) fed by a queue — never blocks the detect/render loops.
- Only fires when InsightFace confirms the person (not on head-only or tracker-only boxes).
- Cadence: every `activity_detect_every = 2.0` seconds per person.
- The most frequent label across all detections in a visit is written to `visits.activity` on close.

---

## 9. Head Detection

- Model: YOLOv8n (ONNX Runtime GPU). Model files in `models/head-yolov8n-onnx/`.
- Supplements face detection to sustain tracking when the face is turned sideways or occluded.
- Always on when the model file exists; gracefully skipped if missing.
- Head bounding boxes are merged with face bounding boxes before the tracking step — a head box that overlaps an existing track keeps the track alive without requiring a face crop.

---

## 10. Hardware Decode (NVDEC)

Controlled by `USE_NVDEC` env var (default **true**). Implementation in `hw_capture.py`.

### `open_capture(source)`

- For RTSP/HTTP sources: tries `HwRtspCapture` (PyAV + CUDA hwaccel) first.
- On any failure: falls back to `cv2.VideoCapture` (software decode via FFmpeg).
- For webcam indices and file paths: always uses `cv2.VideoCapture`.

### `HwRtspCapture`

- Opens the RTSP stream with PyAV, attaching an `HWAccel(device_type="cuda")` context.
- A background thread continuously demuxes and decodes, keeping only the **most recent frame** in memory (one-frame buffer). This matches the `CAP_PROP_BUFFERSIZE=1` behaviour of the existing grid workers.
- `read()` returns and **consumes** the latest frame (returns `False, None` if no new frame since last call).

### Concurrency controls

| Guard | Why |
|-------|-----|
| `_NvdecRWLock` (module-level readers-writer lock) | Prevents SIGSEGV from concurrent NVDEC driver state access. Writers (exclusive): `av.open()` and `container.close()`. Readers (concurrent): `packet.decode()` + `frame.to_ndarray()`. Network I/O (`container.demux()`) runs without any lock. Writers get priority — once a write is queued, new readers block to avoid writer starvation. |
| `_nvdec_transfer_sem` (Semaphore, default **8**) | Caps concurrent `frame.to_ndarray()` GPU→CPU copies — too many simultaneous CUDA memcpy calls can corrupt CUDA state. Redundant with the RW-lock reader path but retained as a secondary throttle. |

`MAX_NVDEC_TRANSFERS` env var overrides the semaphore limit (default **8**, tuned for RTX 5090 with 2 NVDEC engines; lower to 3–4 for laptop/mobile GPUs).

`NVDEC_RECONNECT_CONCURRENCY` limits how many cameras may open a new NVDEC connection simultaneously (default **2**).

`NVDEC_RECONNECT_GAP_SECS` sets the minimum seconds between consecutive NVDEC open/close cycles per capture instance (default **3**).

---

## 11. Database

### Backends

| Backend | How to activate | Notes |
|---------|-----------------|-------|
| SQLite | Default (no config needed) | File path set by `DATABASE_PATH` env var (default `face_recognition.db` in project dir). WAL mode + `wal_autocheckpoint=1000`. |
| PostgreSQL | Set `DATABASE_URL=postgresql://...` | Uses `psycopg2` thread pool (1–5 connections). |

Both backends share the same Python API (`db.py`). Placeholder style differs (`?` for SQLite, `%s` for PostgreSQL) — `db._param()` handles conversion.

### Schema

#### `locations`

Maps a camera source string to a human-readable location name.

| Column | Type | Notes |
|--------|------|-------|
| `id` | integer PK | Auto-increment |
| `camera_source` | text UNIQUE | RTSP URL, webcam index string, etc. |
| `name` | text | Display name (e.g. "Entrance", "Office") |
| `created_at` | timestamp | Row creation time |

#### `sessions`

Each run of the engine (start → stop) is one session.

| Column | Type | Notes |
|--------|------|-------|
| `id` | UUID text | Primary key |
| `started_at` | timestamp | Engine start time |
| `ended_at` | timestamp | Engine stop time (NULL if still running) |
| `camera_source` | text | Primary camera source at session start |

#### `visits`

Core table. One row per continuous presence of a person at a location.

| Column | Type | Notes |
|--------|------|-------|
| `id` | integer PK | Auto-increment |
| `person_name` | text | Matches `faces/<name>/` directory name |
| `location_id` | integer FK | → `locations.id` |
| `first_seen` | timestamp UTC | Visit start |
| `last_seen` | timestamp UTC | Last confirmed detection |
| `ended` | bool/int | 0=open, 1=closed |
| `confidence` | float | Lowest (best) cosine distance in visit |
| `session_id` | UUID FK | → `sessions.id` |
| `screenshot` | text | Face crop filename (served at `/faces/`) |
| `footage` | text | WebM clip filename (served at `/footage/`) |
| `visible_duration` | float | Seconds tracked on camera (footage clock) |
| `activity` | text | Most frequent CLIP action label |
| `branch` | text NOT NULL DEFAULT 'Riyadh' | Branch identifier — auto-assigned from the camera's IP group |

#### `people`

Stores per-person metadata. One row per enrolled person (created on first save; absent until explicitly set).

| Column | Type | Notes |
|--------|------|-------|
| `name` | text PK | Matches `faces/<name>/` directory name |
| `section` | text NOT NULL DEFAULT '' | Name of the section this person belongs to (read-only badge on the People card; managed from the Sections tab) |
| `branch` | text NOT NULL DEFAULT 'Riyadh' | Home branch — determines which branch's absent list this person appears in |
| `email` | text NOT NULL DEFAULT '' | Optional contact email |
| `arabic_name` | text NOT NULL DEFAULT '' | Arabic display name |
| `shift` | text NOT NULL DEFAULT '' | Shift assignment (`'morning'`, `'night'`); '' = unassigned |
| `home_zone_id` | integer FK → `zones.id` (nullable) | The zone this person is expected to be present in |

#### `sections`

Named groups that people can be assigned to (e.g. "IT", "HR"). Managed from Settings → Sections.

| Column | Type | Notes |
|--------|------|-------|
| `id` | integer PK AUTOINCREMENT | Internal ID |
| `name` | text NOT NULL UNIQUE | Section display name |
| `manager` | text NOT NULL DEFAULT '' | Name of the section's manager person |

The `people.section` column is kept in sync: assigning a person to a section sets `people.section = section_name`; unassigning or deleting a section clears it to `''`. Renaming a section (`POST /api/sections/<name>/rename`) updates both the `sections` row and all `people.section` values atomically.

The `manager` column is a soft link to `people.name`; set via `POST /api/sections/<name>/manager`.

#### `zones`

Named camera zones. People can be assigned a home zone; the zone status API reports who is present vs. away from their expected zone.

| Column | Type | Notes |
|--------|------|-------|
| `id` | integer PK AUTOINCREMENT | |
| `name` | text NOT NULL UNIQUE | Zone display name |
| `description` | text NOT NULL DEFAULT '' | Optional description |
| `branch` | text NOT NULL DEFAULT 'Riyadh' | Branch this zone belongs to |
| `created_at` | text DEFAULT datetime('now') | |

#### `zone_cameras`

Many-to-many join between zones and locations (cameras). A zone can cover multiple cameras; a camera can belong to multiple zones.

| Column | Type | Notes |
|--------|------|-------|
| `zone_id` | integer FK → `zones.id` ON DELETE CASCADE | |
| `location_id` | integer FK → `locations.id` ON DELETE CASCADE | |

Primary key: `(zone_id, location_id)`.

#### `gate_events`

Records exit/entry pairs through the designated gate cameras. One row per exit event; the entry timestamp is filled in when the same person returns.

| Column | Type | Notes |
|--------|------|-------|
| `id` | integer PK AUTOINCREMENT | |
| `person_name` | text NOT NULL | |
| `event_date` | text NOT NULL | Calendar date (`YYYY-MM-DD`) of the exit |
| `exit_time` | text NOT NULL | ISO timestamp (UTC) when person hit the exit camera |
| `entry_time` | text | ISO timestamp (UTC) when person returned to the arrival camera; NULL if still out |
| `duration_minutes` | real | `entry_time − exit_time` in minutes; NULL until entry recorded |

Rules:
- A row is opened (`exit_time` set) each time a person is detected on the **exit camera**.
- The most-recent open row for that person is closed (`entry_time` + `duration_minutes`) when they are next detected on the **arrival camera**.
- If a person leaves again before returning, a new row is opened — multiple rows per person per day are normal.
- Rows with `entry_time = NULL` indicate the person is currently out.

#### `daily_reports`

Auto-saved nightly snapshot of the gate report at 23:00.

| Column | Type | Notes |
|--------|------|-------|
| `id` | integer PK AUTOINCREMENT | |
| `report_date` | text UNIQUE NOT NULL | `YYYY-MM-DD` |
| `report_json` | text NOT NULL | Full serialised report (JSON string) |
| `created_at` | text NOT NULL | ISO timestamp when saved |

#### Indexes on `visits`

| Index | Columns | Purpose |
|-------|---------|---------|
| `idx_visits_person` | `person_name` | Person history queries |
| `idx_visits_location` | `location_id` | Location history queries |
| `idx_visits_first_seen` | `first_seen` | Date-range and analytics queries |
| `idx_visits_open` | `person_name, location_id` WHERE `NOT ended` | Fast open-visit lookup per detect cycle |

### Migrations

Additive columns (`screenshot`, `footage`, `visible_duration`, `activity`, `branch`) are added with `ALTER TABLE … ADD COLUMN` at startup, wrapped in try/except so re-running on an already-migrated DB is safe. The `branch` column back-fills all pre-existing rows to `'Riyadh'`.

### Clearing data

```bash
# Flask stopped — direct SQLite:
python3 -c "import db; db.init_db(); db.clear_all_data()"

# Flask running — API:
curl -X POST http://localhost:5001/api/history/clear
```

`clear_all_data()` deletes all visits and sessions. Locations, face images, and footage files are NOT deleted.

---

## 12. Configuration Reference

All settings are read from `.env` (loaded by `python-dotenv` on startup). Copy `.env.example` to get started.

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_PATH` | `face_recognition.db` | SQLite file path |
| `DATABASE_URL` | *(unset)* | PostgreSQL DSN — overrides SQLite |
| `VISIT_TIMEOUT_MINUTES` | `10` | Close a visit after this many minutes without a detection |
| `VISIT_TRANSITION_SECS` | `30.0` | Seconds a person must be absent from their current camera before the visit transitions to a new location |
| `FACE_DETECTION_ENABLED` | `true` | Master AI kill switch. `false` = plain camera stream, no GPU inference |
| `ACTION_DETECTION_ENABLED` | `false` | CLIP zero-shot action classification |
| `AUTO_CAPTURE_ENABLED` | `false` | Auto-save face crops of unknown persons |
| `USE_NVDEC` | `true` | GPU-side RTSP decode via PyAV + CUDA. Set `false` to force CPU decode |
| `MAX_CONCURRENT_RTSP` | `3` | Max cameras with open RTSP connections simultaneously. Raise to your total camera count for all cameras to detect at once. LRU eviction applies when the limit is exceeded. |
| `NVDEC_RECONNECT_CONCURRENCY` | `2` | Max cameras reconnecting to NVDEC simultaneously. Raise to 4+ on RTX 5090. |
| `NVDEC_RECONNECT_GAP_SECS` | `3` | Minimum seconds between consecutive NVDEC open/close cycles per capture instance. |
| `MAX_NVDEC_TRANSFERS` | `8` | Max concurrent GPU→CPU frame copies. Lower to 3–4 for laptop/mobile GPUs; raise on multi-NVDEC GPUs (e.g. RTX 5090). |
| `DETECT_SCALE` | `0.5` | Scale factor applied to frames before face detection. `0.5` = half resolution (faster, less accurate for distant faces). Set to `1.0` for full resolution. Overrides the `detect_scale` passed to `FaceEngine` at startup. |
| `LIVE_ANNOTATIONS_ENABLED` | `true` | Draw bounding boxes on the live MJPEG feed. Footage always annotated. |
| `FOOTAGE_DIR` | *(required)* | Directory where `.mp4` footage clips are written. Recording is skipped if this doesn't resolve to an actual mounted filesystem or has less than `FOOTAGE_MIN_FREE_BYTES` free — see [Footage Recording](#7-footage-recording). |
| `FOOTAGE_MIN_FREE_BYTES` | `1073741824` (1 GiB) | Minimum free bytes required on `FOOTAGE_DIR`'s filesystem before new recordings are allowed |
| `FOOTAGE_MAX_HEIGHT` | `720` | Recorded footage is downscaled to this max height (aspect-ratio preserved) before encoding |
| `FOOTAGE_BITRATE` | *(unset)* | Explicit target video bitrate for the ffmpeg encoder (e.g. `2000000` for 2 Mbps). Unset = let `libx264`'s CRF-equivalent default apply. |
| `FOOTAGE_RETENTION_DAYS` | `7` | Read by `cleanup_footage.sh` (cron) — footage files older than this are deleted |
| `API_KEY` | *(unset)* | Shared-secret key for all `/api/*` routes. Unset = no auth (local dev only) |
| `UI_AUTH_USER` | `admin` | Username for the default admin account seeded into `users.json` on first run (see [UI Login & User Management](#19-ui-login--user-management)) |
| `UI_AUTH_PASSWORD` | `admin123` | Password for that default admin account |
| `FLASK_SECRET_KEY` | *(random per-restart)* | Signs the login session cookie. Set a fixed value so sessions survive a server restart |

### Engine tuning (set in `app.py` at `FaceEngine` instantiation)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `detect_every` | `5` s | Seconds between face detection runs per camera |
| `detect_scale` | `0.5` | Scale factor applied to frames before detection (lower = faster, less accurate). Env-configurable via `DETECT_SCALE`. |
| `width` / `height` | `1280` / `720` | Camera resolution for the MJPEG stream |
| `out_fps` | `15` | Target FPS for MJPEG stream and footage |
| `threshold` | `0.6` | Cosine distance threshold for ArcFace matching (raised from 0.5) |
| `tracker_type` | `CSRT` | OpenCV tracker algorithm (`CSRT` or `KCF`) |
| `INFERENCE_POOL_SIZE` | `6` | Number of parallel InsightFace instances (~300–400 MB VRAM each) |
| `motion_gate` | `true` | Enable CPU motion check before GPU inference |
| `motion_thresh` | `500` | Pixel-diff count threshold for motion gate |
| `viewer_jpeg_quality` | `100` | JPEG quality for the MJPEG viewer stream (separate from detection) |

These can also be changed at runtime without restart via `POST /api/engine/config`.

### Reports config (`reports_config.json`)

| Key | Description |
|-----|-------------|
| `arrival_camera` | Full RTSP URL of the entry/arrival gate camera (cam 15), or the sentinel `"__any__"` (`GATE_ARRIVAL_ANY_CAMERA`) to treat detection on any camera as a valid arrival trigger — see [Gate Events & Daily Reports](#15-gate-events--daily-reports). |
| `exit_camera` | Full RTSP URL of the exit gate camera (cam 16) |
| `manager_email` | Email address for daily report delivery |
| `work_start` | Morning shift start time in `HH:MM` format (e.g. `"09:00"`) |
| `work_end` | Morning shift end time in `HH:MM` format (e.g. `"17:00"`) |
| `late_threshold_minutes` | Minutes after shift start before an arrival is considered late |
| `daily_send_time` | Time to email the daily report (e.g. `"17:00"`) |
| `night_shift_enabled` | Boolean — enable night shift tracking |
| `night_work_start` | Night shift start time in `HH:MM` format (e.g. `"21:00"`) |
| `night_work_end` | Night shift end time in `HH:MM` format (e.g. `"05:00"` next day) |
| `night_late_threshold_minutes` | Minutes after night shift start before a night arrival is late |

Shift times are managed from **Settings → Advanced**. Analytics and gate reports both read from this shared config.

Both `arrival_camera` and `exit_camera` are automatically registered as `high_priority_sources` at engine startup.

---

## 13. File Layout

```
app.py                        Flask server, API routes, visit manager, watchdog
face_engine.py                FaceEngine class — detection, tracking, footage, grid
engine_runner.py              Engine subprocess entry point and command dispatcher
hw_capture.py                 NVDEC-accelerated RTSP capture (cv2.VideoCapture-compatible)
head_detector.py              YOLOv8n head detector wrapper (ONNX Runtime GPU)
action_detector.py            CLIP ViT-B/32 zero-shot action classifier (ONNX Runtime GPU)
db.py                         Database layer (SQLite / PostgreSQL)

templates/
  index.html                  Main dashboard (live feed, cameras, analytics, attendance)
  history.html                Visit history dashboard (daily, per-person, per-location)
  people.html                 People management (view, rename, delete, transfer)
  tracker.html                Camera Tracker page (line-crossing setup + live event feed)

faces/                        Enrolled face images — one subdirectory per person
  <name>/
    photo.jpg
    .arcface.npz              Cached ArcFace embeddings (auto-generated, do not commit)

footage/                      Recorded visit MP4 clips (or FOOTAGE_DIR mount)
models/
  head-yolov8n-onnx/          YOLOv8n head detector ONNX model (~12 MB)
  clip-vit-base-patch32-onnx/ CLIP ViT-B/32 ONNX model (~600 MB)

grid_config.json              Saved grid layout + camera slot assignments
users.json                    UI login accounts (username, password, locked_branch, can_manual_attendance, is_admin) — auto-created with default admin/mustafa accounts if missing
face_recognition.db           SQLite database (default, not committed)
docker-compose.yml            Optional PostgreSQL via Docker
cleanup_footage.sh            Cron script: deletes footage older than FOOTAGE_RETENTION_DAYS

.env                          Runtime configuration (not committed)
.env.example                  Example configuration template
api.md                        External API reference
documentation.md              This file
README.md                     Setup, features, and troubleshooting guide
```

---

## 14. Web UI Pages

### `/` — Main Dashboard

- **Branch switcher**: Riyadh / Egypt tabs at the top of the page. The active branch is persisted in `localStorage`. All history and analytics requests include `?branch=<active>` so each tab shows data for its own location.
- **Live feed**: MJPEG stream from `/video`. Auto-reconnects on error; retries every 2 s after engine recovery.
- **Camera controls**: carousel arrows / swipe to cycle cameras. The dashboard viewer is **single-camera-only** now — the old Single/Multi mode toggle and the grid-layout picker/dropdown were removed from this page. The analysis pool still covers every configured camera regardless of what the viewer shows; grid-layout viewing is still reachable via direct `POST /api/camera` calls with a `grid_RxC` source, just not from this UI.
- **Analytics tab** (default): Arrivals by shift (single table per shift with an Earliest/Latest toggle; both datasets fetched in parallel on load and cached so toggling is instant), Top 10 Longest Working bar chart, Daily Headcount bar chart, Attendance Heatmap.
- **Attendance tab**: real-time roster driven by SSE stream (`/api/attendance/stream`).
- **Add Manual tab**: see [Manual Attendance](#18-manual-attendance).

### `/history` — Visit History

Four sub-views selectable by tab:
- **Attendance** — first/last seen per person per day.
- **Daily Summary** — all visits on a selected date.
- **Per Person** — all visits for a selected person over a date range.
- **Per Location** — all visits at a selected location over a date range.

All date pickers use `dd-mm-yyyy` display (Flatpickr with `altInput`).

### `/settings` — Settings

The settings page is organised into tabs:

| Tab | Description |
|-----|-------------|
| **People** | Enrol new people, view/rename/delete enrolled persons, manage face images |
| **Sections** | Create and manage named groups (e.g. "IT", "HR") with section managers |
| **Managers** | Assign manager/contact email addresses to each person |
| **Camera** | Configure IP camera groups, RTSP URLs, and channel numbers |
| **Zones** | Define zones, assign cameras (shown by display name), set away thresholds |
| **Reports** | Configure arrival/exit cameras, manager email, daily auto-send time; generate gate reports |
| **Advanced** | Configure shift start/end times and late thresholds for analytics and reports |

### `/people` — People Management (Settings)

- View all enrolled persons with face thumbnails and image count.
- Each card shows a read-only **section badge** (set from the Sections tab) and a **branch** dropdown (Riyadh / Egypt) — branch changes save instantly to the `people` DB table.
- Rename (moves `faces/` directory and updates all `visits.person_name` rows and the `people` row).
- Delete (removes `faces/` directory and all visit and people rows).
- View individual face images; delete single images; transfer images between persons.
- **Set gallery thumbnail**: a star button on each image in the Face Images modal makes that image the person's thumbnail. Implemented in `POST /api/person/<name>/image/<filename>/set_thumbnail` by swapping filenames so the chosen image occupies the lowest numeric slot (`list_people()` always uses the numerically-lowest image as the thumbnail) — no separate thumbnail field or DB column. `GET /api/person/<name>/images` uses the same numeric-aware sort as `list_people()` so the starred image in the gallery always matches what actually renders as the thumbnail.
- Bulk delete / bulk transfer.
- Merge multiple persons into one.

### `/sections` — Sections Management (Settings)

- Create named sections (e.g. "IT", "HR", "Security").
- Each section card lists its current members.
- Assign any enrolled person to a section via the assign button; this replaces their current section.
- Remove a person from a section (unassign), or delete the entire section (clears all members' section field).
- Section membership is reflected as a read-only badge on each person's People card.

---

## 15. Gate Events & Daily Reports

### Gate event flow

The `_handle_gate_event()` function in `app.py` is called from `_update_visit_for_person()` at two points:
1. When a new visit is opened for a person (initial detection on any camera).
2. When a person transitions from one camera to another.

If the camera is the **exit camera**: `db.open_gate_exit()` writes a new `gate_events` row with `exit_time = now`.
If the camera is the **arrival camera**: `db.close_gate_entry()` finds the most-recent open row for that person and fills in `entry_time` + `duration_minutes`. If there is no open exit row to close (e.g. the exit camera is offline), `db.write_arrival_event()` writes a standalone row (`exit_time == entry_time`) so the arrival still shows up in reports. This standalone write is skipped if an open exit row already exists, or if a standalone arrival was already written for that person within the last 30 minutes (debounce).

The gate camera URLs are cached for 60 seconds (`_gate_camera_cache`) to avoid re-reading `reports_config.json` on every visit transition.

### "Any camera" arrival trigger (`GATE_ARRIVAL_ANY_CAMERA`)

`reports_config.json`'s `arrival_camera` can be set to the sentinel value `"__any__"` (constant `GATE_ARRIVAL_ANY_CAMERA` in `app.py`) instead of a single camera's RTSP URL. When set, `_handle_gate_event()` treats detection on **any** camera as a valid arrival trigger for closing an open exit event — useful when there's no single fixed entry point a person always passes through. Settings → Reports exposes this as an "All cameras (any camera counts as arrival)" option in the Arrival Camera dropdown. `_gate_camera_display_name()` resolves the report header's camera-name label for this case (`"Any camera"`, or `"Any camera / <exit camera name>"` when an exit camera is also configured), since the sentinel has no matching row in the `locations` table to look up a display name from.

### Daily report

`_daily_report_scheduler_loop()` runs as a background thread and saves the gate report at **23:00** local time every day. The report is persisted to the `daily_reports` table.

### Gate report content

`_generate_gate_report()` returns one row per enrolled person per day, including absent people. Columns: `name`, `arrival` (earliest visit on ANY camera), `exits` (list of exit/entry pairs from `gate_events`), `status` (on time / late / absent).

`unknown_N` auto-captured persons are excluded entirely from the report — both from the raw `gate_events` query and from the people-meta query used to build the absent list — since they're unidentified tracks, not real employees worth reporting on.

Standalone-arrival rows (written by `write_arrival_event()` when there's no open exit event to pair with, where `exit_time == entry_time`) are filtered out of the report's exit-events list. They aren't real exits and previously rendered as bogus "0-minute" exit/return pairs; the arrival time they carry still surfaces via the separate arrival-time lookup (earliest visit on any camera), so no information is lost by dropping them from the exits list.

---

## 16. Analytics & Reports

All analytics endpoints live under `/api/analytics/`. Dates use the server's **local timezone** for shift boundaries; stored timestamps are UTC and converted on query.

All analytics (and history) endpoints accept an optional `?branch=<name>` query parameter (e.g. `?branch=Riyadh` or `?branch=Egypt`). When omitted, data for all branches is returned. The in-tree UI always appends the active branch tab's value.

### Present / Absent Name Lists (`/api/analytics/present_absent`)

Returns the full name lists behind the Present and Absent tiles. Called lazily on tile click rather than on page load.

- `present` — known persons with at least one visit on the given day (branch-filtered when `?branch=` is provided), sorted alphabetically.
- `absent` — persons assigned to this branch in the `people` table with no visit that day, sorted alphabetically. Without a branch filter, falls back to all enrolled known `faces/` folders with no visit that day.

The Present and Absent tiles have a hover border effect (emerald / rose) and open a modal with the name list and count on click. The modal closes on backdrop click or Escape.

### Summary Tiles (`/api/analytics/summary`)

A single endpoint that returns three KPIs for a given day, loaded in one request to populate the stat tile row at the top of the Analytics tab:

| Field | Description |
|-------|-------------|
| `peak_hour` | Local-time hour bucket with the most distinct people spotted, e.g. `"09:00 – 10:00"`. `null` if no visits that day. |
| `present_today` | Count of distinct known persons (non-`unknown_N`) with at least one visit today (branch-filtered). |
| `absent_today` | Count of persons assigned to this branch in the `people` table with no visit today. Without a branch filter, counts all enrolled known folders minus present. |
| `unknowns_today` | Count of `unknown_N` folders currently in `faces/` — total unresolved auto-captured persons in the system, regardless of when they were last seen. |

### Earliest / Latest Arrivals (`/api/analytics/earliest`)

- Returns top 10 persons by first arrival time on a given date.
- `&order=latest` reverses to latest arrivals.
- `&shift=morning` — window: `work_start - 1h` → `work_end` (local time). Both boundaries are configurable from **Settings → Advanced**.
- `&shift=night` — window: `night_work_start - 1h` (today) → `night_work_end` (next day, local time). Automatically excludes anyone who already appeared in the morning window, so each person is listed in at most one shift.
- The UI shows one table per shift. Both earliest and latest datasets are fetched in parallel on page load and cached in memory; the Earliest/Latest toggle switches between them instantly without a new request.

### Top 10 Longest Working (`/api/analytics/longest`)

- Periods: `day`, `week` (Sun–Sat), `month` (1st–last), `year` (Jan–Dec).
- Uses `visible_duration` when available (actual on-camera seconds), falls back to `last_seen − first_seen`.
- Rendered as a Chart.js horizontal bar chart. Hover: hovered bar turns white, all others stay green.

### Daily Headcount (`/api/analytics/headcount`)

- Returns distinct person count per calendar day over a date range (default: current month).
- Excludes `unknown_N` names.
- Rendered as a Chart.js vertical bar chart.

### Attendance Heatmap (`/api/analytics/heatmap`)

- Returns a person × day presence matrix over a date range (default: current month).
- Rendered as a scrollable HTML table (max-height 400 px) with emerald cells for present days.
- Excludes `unknown_N` names.

### Gate report export (Excel / PDF)

The **Reports** tab in Settings can export the currently displayed gate report:

- **Excel** (`exportReportExcel()`): builds a UTF-8 BOM-prefixed CSV client-side (title row, header row, one row per person) and triggers a browser download. Opens correctly in Excel despite the `.csv` extension.
- **PDF** (`exportReportPDF()`): renders an HTML table into a new browser tab/window and calls `window.print()`, so the "export" is really "print to PDF" via the browser's print dialog.

Both respect whatever filter (`present`/`absent`/`late`/`ontime`) and shift filter are currently applied to the report table — they do not re-fetch, they export what's on screen.

### Report History modal

The **Report History** section in Settings → Reports no longer has a manual refresh button. Its **View** button opens a modal (`openReportHistoryListModal()`) that lazy-loads the saved-report list from `GET /api/reports/history` only when opened, rather than fetching on every page load. Each row in that list has its own **View** button which fetches `GET /api/reports/history/<date>` and opens a second modal, re-parenting the existing report-results card into a modal container (`reportHistoryModalSlot`) to render that date's full detail report without duplicating the rendering logic.

---

## 17. Camera Tracker (Line-Crossing)

A dedicated page (`/tracker`) for counting people crossing a configurable line on up to **4** pinned cameras — independent of the visit/attendance system.

### Setup page (`templates/tracker.html`)

- 2×2 grid of camera tiles. Each tile supports pan/zoom/rotate (drag to pan, scroll/pinch to zoom, stored per-camera) and an optional rectangular ROI — when set, only track centers inside the ROI are considered for crossing detection.
- A draggable horizontal line per camera (`line_y_ratio`, expressed relative to the ROI if one is set, otherwise relative to the full tile) marks the crossing boundary.
- A live event feed (SSE) shows crossings as they happen, with the annotated snapshot.

### Config (`tracker_config.json`, via `/api/tracker/config`)

| Key | Description |
|-----|--------------|
| `cameras` | List of up to 4 RTSP URLs assigned to tracker tiles |
| `line_y_ratio` | Legacy/global default line position (0.05–0.95) |
| `line_y_ratios` | Per-camera line position `{source: ratio}` |
| `cam_transforms` | Per-camera `{zoom, panX, panY, rotate}` view transform, purely cosmetic (does not affect detection coordinates) |
| `tracker_rois` | Per-camera normalized ROI `[rx, ry, rw, rh]` (0–1 range); crossing detection ignores tracks outside it |

Cameras assigned here are automatically **pinned** in the RTSP pool (immune to `MAX_CONCURRENT_RTSP` LRU eviction) and added to `high_priority_sources` (full-res, no motion gate) via `FaceEngine.set_config()`.

### Crossing detection (`face_engine.py`, `_grid_detect_loop`)

For each track on a tracker-enabled camera:
1. Compute the track's bbox-center Y position, normalized to the tile (or to the ROI, if one is set for that camera).
2. Maintain a 3-frame history of which side of the line the center falls on (`side_history`), keyed by `(camera_source, _tid)` where `_tid` is a per-worker monotonic track ID (stable across frames, unlike `id(track)`).
3. A crossing fires only when all 3 history entries agree on the new side **and** it differs from the last confirmed side **and** at least `3.0` s (debounce) have passed since the last crossing for that track.
4. On a confirmed crossing, an event (`camera_source`, `tid`, `name`, `direction` = `enter`/`exit`, `bbox`, a copy of the raw frame) is pushed onto an in-memory queue (`_tracker_crossing_queue`, max 100).
5. Track state is pruned once a track disappears from the live track list.

### Event pipeline

`engine_runner.py` drains `pop_crossing_events()` every poll cycle, draws the bbox onto a copy of the raw frame, saves it as a JPEG under `static/tracker_snapshots/<uuid>.jpg`, and mirrors the serializable event (with `snapshot_path`, minus the raw frame) into the shared `Manager` dict (`tracker_crossing_events`, capped at the last 50).

`app.py`'s `_tracker_poll_loop()` thread polls `engine.pop_tracker_crossing_events()` every 0.5 s, persists each event via `db.insert_tracker_event()`, and pushes it onto `tracker_events_q` for the `/api/tracker/stream` SSE endpoint.

### Database: `tracker_events` table

| Column | Type | Notes |
|--------|------|-------|
| `id` | integer PK | |
| `event_type` | text | `enter` or `exit` |
| `person_name` | text | Recognised name, or `"unknown"` |
| `camera_source` | text | RTSP URL |
| `camera_name` | text | Resolved display name at event time |
| `occurred_at` | timestamp | |
| `snapshot_path` | text (nullable) | Relative path under `static/tracker_snapshots/` |
| `confidence` | float | Track's best cosine distance at crossing time |

Indexed on `occurred_at DESC` and `person_name`. `db.delete_tracker_events_before(cutoff_iso)` is available for retention cleanup but is not currently wired to a scheduler.

### API

| Method | Path | Description |
|--------|------|--------------|
| GET | `/tracker` | Tracker setup + live feed page |
| GET | `/api/tracker/config` | Current tracker camera assignment, line positions, transforms, ROIs |
| POST | `/api/tracker/config` | Update the above; validates camera sources against configured IP cameras |
| GET | `/api/tracker/events?limit=&offset=` | Paginated event history (max 500/page) |
| POST | `/api/tracker/ping` | Keep-alive from the tracker page — see bandwidth note below |
| GET | `/api/tracker/stream` | SSE stream of new crossing events (`event: crossing`) plus an initial `event: snapshot` backfill of the last 20 |

### Bandwidth interaction with the main dashboard

While the tracker page is actively pinging (`POST /api/tracker/ping`, tracked via `_tracker_active_t` with an 8 s TTL), the main dashboard's composite MJPEG stream (`mjpeg_generator`) throttles itself to 1 fps so bandwidth is freed for the tracker page's per-camera streams. Tracker-assigned camera streams (`/video/<source>`) also get a much longer "No Signal" grace period (30 s vs 2 s) since they are pinned/high-priority and briefer reconnects shouldn't flash a placeholder.

---

## 18. Manual Attendance

An **Add Manual** tab on the main dashboard's Attendance section lets an admin backfill attendance for a day without a camera detection — e.g. for someone who worked off-site.

- UI: pick a date, an arrival time (default `09:00`) and a departure time (default `17:00`), then check off any subset of enrolled people from a grid (Select All / Deselect All helpers), and submit.
- `POST /api/attendance/manual` — body `{date: "YYYY-MM-DD", arrived: "HH:MM", left: "HH:MM", names: [...]}`. For each name, calls `db.insert_manual_visit(name, first_seen, last_seen)`.
- `db.insert_manual_visit()` inserts a `visits` row directly in the **closed** state (`ended = true`), with `location_id = NULL` (no camera association), `confidence = 1.0`, and `branch` defaulting to `"Riyadh"`. It shows up in history/analytics like any other visit.

---

## 19. UI Login & User Management

Session-based login gates the browser-facing pages and API — it has no effect on the camera engine subprocess itself, which captures, detects, and records footage regardless of whether anyone is logged into the web UI. This is a separate system from the `API_KEY` external-request gating described in [api.md](api.md#2-authentication) — `API_KEY` protects `/api/*` for external/API clients (header or query string), while UI login protects the browser session (cookie-based, via Flask's `session`).

### Users config (`users.json`)

Users are persisted in `users.json` (`_USERS_CONFIG_PATH` in `app.py`), loaded into the in-memory `UI_USERS` dict at startup and rewritten on every change via `_save_users()`. If the file doesn't exist, it's seeded from `_DEFAULT_USERS`: an `admin` account (username/password from `UI_AUTH_USER`/`UI_AUTH_PASSWORD` env vars, defaulting to `admin`/`admin123`) with `is_admin = True`, and a `mustafa` account locked to the `Egypt` branch with `can_manual_attendance = False`. Passwords are stored in **plaintext** in `users.json` — acceptable for this app's threat model (trusted operators, not a public multi-tenant service) but worth revisiting if that changes.

Each user record has:

| Field | Type | Description |
|-------|------|-------------|
| `password` | string | Plaintext password |
| `locked_branch` | string or `null` | If set, every branch-filterable request from this user is forced server-side to this branch — see below. `null` for admins and unrestricted users. |
| `can_manual_attendance` | bool | Whether this user may call `POST /api/attendance/manual`. |
| `is_admin` | bool | Grants access to `/api/users` (User Management) and forces `locked_branch = null`. |

### Login / logout

- `GET/POST /login` — renders (GET) or processes (POST) the login form. On success, clears and repopulates the Flask `session` with `username` and marks it permanent. Redirects to `next` (or `/`) on success; re-renders the form with an error on failure.
- `POST /logout` — clears the session and redirects to `/login`.

### `before_request` enforcement (`_require_ui_auth`)

Runs on every request except `/login`, `/logout`, and `/static/*`. If no user is logged in: API/stream routes get a `401 {"ok": false, "error": "unauthorized"}` JSON response, page routes redirect to `/login?next=<path>`.

For a logged-in user:

- If `can_manual_attendance` is `False`, `POST /api/attendance/manual` is rejected with `403`.
- If `locked_branch` is set, it is enforced in several ways — **server-side, not just hidden in the UI**, so it can't be bypassed by calling the API directly:
  - Any request's `branch` query parameter is overridden to `locked_branch`, even if the request didn't send one at all (several endpoints treat a missing `branch` as "no filter, all branches" — leaving that case unhandled would leak cross-branch data by omission).
  - `/api/ip_cameras/groups/<id>...` and `/api/ip_cameras/cameras/<id>...` routes are blocked with `403` if the targeted group's `branch` doesn't match `locked_branch` — covers viewing/testing, mutating cameras, and group update/delete/reorder.
  - `/api/zones/<id>...` routes are blocked with `403` if the targeted zone's `branch` doesn't match `locked_branch`.
  - The camera device list (`/api/camera` devices) and IP camera group list (`/api/ip_cameras`) are filtered server-side to only that branch.

### Admin-only User Management API

Gated by `_require_admin()` (checks `current_user().is_admin`), returning `403 {"ok": false, "error": "admin only"}` otherwise.

| Method | Path | Description |
|--------|------|--------------|
| GET | `/api/users` | List all users: `{username, locked_branch, can_manual_attendance, is_admin}`. |
| POST | `/api/users` | Create a user. Body: `{username, password, is_admin?, locked_branch?, can_manual_attendance?}`. `locked_branch` is forced to `null` when `is_admin` is true. 400 if username/password missing or username already exists. |
| PUT | `/api/users/<username>` | Update a user. Body: any subset of `{password, is_admin, locked_branch, can_manual_attendance}`. Refuses (`400`) to demote the last remaining admin — otherwise nobody could reach User Management to fix it. |
| DELETE | `/api/users/<username>` | Delete a user. Refuses (`400`) to delete the last remaining admin, or to delete the account you're currently logged in as. |

### Settings → User Management tab

Admin-only tab (hidden entirely for non-admin users, both the sidebar entry and the tab content, gated by the `is_admin` template flag). Lists all users with their branch lock and manual-attendance permission; a modal lets an admin create or edit a user (username, password, admin toggle, locked branch dropdown — disabled when "admin" is checked since admins always see all branches — and a manual-attendance checkbox).

---

## 20. Manual Detect

A **Detect** button on the main dashboard's live feed (`templates/index.html`) lets a user trigger an immediate, one-shot face/head detection pass on the camera currently being viewed — bypassing both `AUTO_CAPTURE_ENABLED` and the normal multi-second auto-capture accumulation window described in [Auto-Capture](#6-auto-capture-unknown-persons). Useful for immediately enrolling or marking attendance for someone on camera right now, without waiting for the automatic pipeline's timing.

### Flow

1. **`POST /api/camera/manual_detect`** (body `{source}`) calls `FaceEngine.manual_detect(camera_source)`, which:
   - Freezes the camera's `latest_raw_frame` (the same per-worker frame buffer the live grid capture loop maintains).
   - Runs `_detect_and_match_faces(..., force_full_res=True)` and `_run_head_detection()` on the frozen frame.
   - For each detected face already matching a known person, reports the match. For an unmatched ("unknown") face, immediately saves it as a new `faces/unknown_N/1.jpg` via `_save_manual_unknown()` — the same crop/padding/size-gate logic as the auto-capture path (`_try_capture_more_for_known`'s sibling), but skipping the multi-second accumulation window and the `auto_capture_enabled` gate, since the manual click itself is the "capture this" signal a human would otherwise have to wait several seconds for the automatic pipeline to infer.
   - Saves the frozen frame as a JPEG under `static/tracker_snapshots/manual_detect_<uuid>.jpg` and returns its path (see "Frame hand-off" below).
   - Returns `{ok, error, faces: [{bbox, name, confidence, captured}], heads: [{bbox, confidence}], heads_detected, frame_path, frame_width, frame_height}`.
   - Back in `app.py`, every matched (or newly-captured) face is also fed through `_update_attendance_from_tracks()` — the same function the live detection loop uses — via synthetic "fake track" dicts, so a manual Detect has the same real-world effect on visits/gate-events as being seen by the automatic pipeline, not just a passive lookup.
2. The UI opens a modal showing the frozen frame with detection boxes drawn on top: **green** boxes for recognized/captured faces (with a name label), **amber** for the hovered/selected box, **blue** for head-only detections with no matching face.
3. **Click a face box** → assign it to an existing or a new person via the existing `POST /api/rename_person` route (renames the auto-captured `unknown_N` folder to the chosen name).
4. **Click a head box with no matching face** → the UI first tries `POST /api/camera/manual_detect_region` (body `{frame_path, x, y, w, h}`), which calls `FaceEngine.manual_detect_region()`: re-runs face + head detection on just that region (with padding and a smaller `min_face_size`) of the already-frozen frame, since a face too small to clear the full-frame detection threshold often clears it once examined at a higher effective resolution. If that still finds no face, the UI falls back to `POST /api/camera/assign_head_crop` (body `{frame_path, x, y, w, h, person_name}`), which calls `FaceEngine.assign_head_crop()` to save the head crop itself as a reference photo in that person's `faces/` folder. This does **not** contribute a face embedding — InsightFace cannot extract one from a crop with no visible face — so `reload_faces()` simply skips the image during matching, exactly as it already skips any other unembeddable photo. It exists purely so a human who can identify someone from body shape/position/context (something no model here does) has a way to record that identification, and so the image shows up in that person's gallery.

### Frame hand-off: saved file, not a Pipe blob

The frozen frame is handed from the engine subprocess to Flask via a file saved under `static/tracker_snapshots/`, with the path returned to the client — not as a base64-encoded blob sent back over the `multiprocessing.Pipe`. A 4K JPEG can be several hundred KB to a few MB as base64, and a real bug was found where one bad multi-MB message on the pipe intermittently came back as a `None` result on the Flask side — a pipe-level issue with no corresponding exception logged in the engine subprocess. The [Camera Tracker](#17-camera-tracker-line-crossing)'s crossing-event snapshots use the same disk-handoff pattern for the same reason.

`_validate_manual_detect_frame_path()` in `app.py` restricts `frame_path` (used by both `manual_detect_region` and `assign_head_crop`) to files directly under `static/tracker_snapshots/` whose basename starts with `manual_detect_`, rejecting anything else — this prevents the endpoint from being used to read arbitrary files off disk via a crafted `frame_path`.
