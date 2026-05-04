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
15. [Analytics & Reports](#15-analytics--reports)

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

### Inter-process communication

| Channel | Direction | Used for |
|---------|-----------|----------|
| `multiprocessing.Pipe` | bidirectional | Commands (Flask → engine) and responses |
| `multiprocessing.Manager` dict | engine writes, Flask reads | Live JPEG, tracks, status, fps |

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

### Recognition matching

- Each detected face embedding is compared against all known person templates using **cosine distance**.
- Match threshold: `0.4` (default in `FaceEngine.__init__`; `app.py` passes `0.5`).
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

### IP camera groups

- Configured via the UI under Settings → IP Cameras.
- Stored as groups (e.g. "Floor 1") containing cameras with a name and RTSP URL or channel number.
- `POST /api/ip_cameras/groups/<id>/cameras` accepts `{name, channel}` (builds URL from group `base_url`) or `{name, url}` (explicit RTSP URL).

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
| `footage` | text | Filename of WebM footage clip |
| `visible_duration` | float | Actual seconds on camera (footage writer clock) |
| `activity` | text | Most frequent CLIP action label during visit |

### `duration_secs` vs `visible_duration`

- `duration_secs` (serialised in API responses) = `last_seen − first_seen` — wall-clock elapsed time from first to last detection. Can be long if the person was mostly off-camera but briefly detected again.
- `visible_duration` = actual seconds the footage writer was running (i.e., seconds the person was continuously tracked). Used by the Top 10 Longest Working analytics endpoint.

---

## 6. Auto-Capture (Unknown Persons)

Controlled by `AUTO_CAPTURE_ENABLED` env var (default **false**).

### Enrolment flow

1. An unrecognised face is tracked. The engine accumulates frames and keeps the highest-quality crop (largest bounding box area).
2. After **5 continuous seconds** of tracking (`_unknown_capture_min_seconds`), the best crop is quality-gated:
   - InsightFace must re-detect a face in the crop with `det_score ≥ 0.50`.
   - Crop dimensions must be ≥ 60×60 px.
   - Passes a deduplication check against all existing `unknown_N` embeddings (cosine distance > 0.5 required to confirm it's a new person).
3. On pass: image saved to `faces/unknown_N/1.jpg`; track promoted to `unknown_N`.

### Bootstrap accumulation

- For the first **10 images** after enrolment, additional samples are saved with only a **5-second cooldown** per camera (instead of the steady-state 10 minutes).
- This quickly builds a diverse averaged template, fixing the "single-frame template never matches re-appearances" failure mode.

### Demotion

- Tracks that lose recognition for **12 consecutive frames** (~3.6 s at default cadence) are demoted from `unknown_N` back to "unknown" and become eligible for new enrolment.

### Limits

- Maximum **30 images** per person.
- Maximum **150** auto-captured `unknown_N` folders total.

---

## 7. Footage Recording

- Every open visit with a known person (not raw "unknown") starts a `cv2.VideoWriter` writing VP8/WebM.
- Frames come from a **ring buffer** (`FOOTAGE_RING_SECS = 1.0 s`) so the clip starts slightly before the visit opened.
- Footage is written to `FOOTAGE_DIR` (required env var). Can be a NAS mount path.
- On `close_visit()`, the writer is flushed and released; `visible_duration` is written to the DB.
- Footage files are served at `/footage/<filename>` (key-authenticated).
- Footage clips are **always annotated** (bounding boxes + labels) regardless of `LIVE_ANNOTATIONS_ENABLED`.

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
| `_nvdec_lock` (module-level `threading.Lock`) | Serialises `av.open()` and `container.close()` — simultaneous NVDEC context creation/destruction causes SIGSEGV |
| `_nvdec_transfer_sem` (Semaphore, default **6**) | Caps concurrent `frame.to_ndarray()` GPU→CPU copies — too many simultaneous CUDA memcpy calls corrupt CUDA state |

`MAX_NVDEC_TRANSFERS` env var overrides the semaphore limit (useful on GPUs with multiple NVDEC engines like RTX 5090).

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

#### Indexes on `visits`

| Index | Columns | Purpose |
|-------|---------|---------|
| `idx_visits_person` | `person_name` | Person history queries |
| `idx_visits_location` | `location_id` | Location history queries |
| `idx_visits_first_seen` | `first_seen` | Date-range and analytics queries |
| `idx_visits_open` | `person_name, location_id` WHERE `NOT ended` | Fast open-visit lookup per detect cycle |

### Migrations

Additive columns (`screenshot`, `footage`, `visible_duration`, `activity`) are added with `ALTER TABLE … ADD COLUMN` at startup, wrapped in try/except so re-running on an already-migrated DB is safe.

### Clearing data

```bash
# Flask stopped — direct SQLite:
python3 -c "import db; db.init_db(); db.clear_all_data()"

# Flask running — API:
curl -X POST http://localhost:5000/api/history/clear
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
| `MAX_NVDEC_TRANSFERS` | `6` | Max concurrent GPU→CPU frame copies (raise on multi-NVDEC GPUs like RTX 5090) |
| `LIVE_ANNOTATIONS_ENABLED` | `true` | Draw bounding boxes on the live MJPEG feed. Footage always annotated. |
| `FOOTAGE_DIR` | *(required)* | Directory where WebM footage clips are written |
| `API_KEY` | *(unset)* | Shared-secret key for all `/api/*` routes. Unset = no auth (local dev only) |

### Engine tuning (set in `app.py` at `FaceEngine` instantiation)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `detect_every` | `5` s | Seconds between face detection runs per camera |
| `detect_scale` | `1.0` | Scale factor applied to frames before detection (lower = faster, less accurate) |
| `width` / `height` | `1280` / `720` | Camera resolution for the MJPEG stream |
| `out_fps` | `15` | Target FPS for MJPEG stream and footage |
| `threshold` | `0.5` | Cosine distance threshold for ArcFace matching |
| `tracker_type` | `CSRT` | OpenCV tracker algorithm (`CSRT` or `KCF`) |
| `INFERENCE_POOL_SIZE` | `6` | Number of parallel InsightFace instances (~300–400 MB VRAM each) |

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

faces/                        Enrolled face images — one subdirectory per person
  <name>/
    photo.jpg
    .arcface.npz              Cached ArcFace embeddings (auto-generated, do not commit)

footage/                      Recorded visit WebM clips (or FOOTAGE_DIR mount)
models/
  head-yolov8n-onnx/          YOLOv8n head detector ONNX model (~12 MB)
  clip-vit-base-patch32-onnx/ CLIP ViT-B/32 ONNX model (~600 MB)

grid_config.json              Saved grid layout + camera slot assignments
face_recognition.db           SQLite database (default, not committed)
docker-compose.yml            Optional PostgreSQL via Docker

.env                          Runtime configuration (not committed)
.env.example                  Example configuration template
api.md                        External API reference
documentation.md              This file
README.md                     Setup, features, and troubleshooting guide
```

---

## 14. Web UI Pages

### `/` — Main Dashboard

- **Live feed**: MJPEG stream from `/video`. Auto-reconnects on error; retries every 2 s after engine recovery.
- **Camera controls**: carousel arrows / swipe to cycle cameras in single mode; grid layout selector.
- **Analytics tab** (default): Arrivals by shift (single table per shift with an Earliest/Latest toggle; both datasets fetched in parallel on load and cached so toggling is instant), Top 10 Longest Working bar chart, Daily Headcount bar chart, Attendance Heatmap.
- **Attendance tab**: real-time roster driven by SSE stream (`/api/attendance/stream`).

### `/history` — Visit History

Four sub-views selectable by tab:
- **Attendance** — first/last seen per person per day.
- **Daily Summary** — all visits on a selected date.
- **Per Person** — all visits for a selected person over a date range.
- **Per Location** — all visits at a selected location over a date range.

All date pickers use `dd-mm-yyyy` display (Flatpickr with `altInput`).

### `/people` — People Management

- View all enrolled persons with face thumbnails and image count.
- Rename (moves `faces/` directory and updates all `visits.person_name` rows).
- Delete (removes `faces/` directory and all visit rows).
- View individual face images; delete single images; transfer images between persons.
- Bulk delete / bulk transfer.
- Merge multiple persons into one.

---

## 15. Analytics & Reports

All analytics endpoints live under `/api/analytics/`. Dates use the server's **local timezone** for shift boundaries; stored timestamps are UTC and converted on query.

### Summary Tiles (`/api/analytics/summary`)

A single endpoint that returns three KPIs for a given day, loaded in one request to populate the stat tile row at the top of the Analytics tab:

| Field | Description |
|-------|-------------|
| `peak_hour` | Local-time hour bucket with the most distinct people spotted, e.g. `"09:00 – 10:00"`. `null` if no visits that day. |
| `present_today` | Count of distinct known persons (non-`unknown_N`) with at least one visit. |
| `unknowns_today` | Count of `unknown_N` folders currently in `faces/` — total unresolved auto-captured persons in the system, regardless of when they were last seen. |

### Earliest / Latest Arrivals (`/api/analytics/earliest`)

- Returns top 10 persons by first arrival time on a given date.
- `&order=latest` reverses to latest arrivals.
- `&shift=morning` — window: 04:00–16:00 local time.
- `&shift=night` — window: 16:00–04:00 local time (next day). Automatically excludes anyone who already appeared in the morning window, so each person is listed in at most one shift.
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
