
# Face Recognition Attendance System

A Flask-based face recognition and attendance system with multi-camera support, visit tracking, footage recording, CLIP-based action detection, YOLO head detection, and auto-capture of unknown persons.

Uses InsightFace (RetinaFace + ArcFace, ONNX Runtime GPU) for face detection and recognition, OpenCV for tracking, CLIP (ONNX Runtime) for zero-shot action classification, and YOLOv8n for head detection.

## Branches

| Branch | Target | Notes |
|--------|--------|-------|
| `main` | CPU (Windows / Ubuntu) | No GPU acceleration |
| `ubuntu-dev` | **GPU (Ubuntu + NVIDIA)** | CUDA + ONNX Runtime GPU. **Do not merge into main.** |

## Features

- **Multi-camera analysis** — every configured camera runs concurrently in a shared analysis pool: face detection, recognition, tracking, visit bookkeeping, and footage capture all happen for every camera at once regardless of which one is on screen. RTSP streams decode on the GPU's NVDEC engine when available (PyAV + CUDA hwaccel), keeping CPU free for the rest of the pipeline. Per-camera fallback to software decode if hardware decode init fails.
- **UI is single-camera-only** — the live preview cycles between cameras one at a time (carousel arrows / swipe). Bounding boxes and name labels are drawn on the live feed by default; toggle off via `LIVE_ANNOTATIONS_ENABLED=0` when crowded scenes turn the overlays into illegible mess. Footage recordings are always annotated regardless of this flag.
- **Face recognition** — InsightFace `buffalo_l` model pack (RetinaFace detector + ArcFace embeddings, ONNX Runtime on GPU) with CSRT tracking. Thread-safe so multiple cameras detect concurrently in a single shared FaceAnalysis instance.
- **Head detection** — YOLOv8n (ONNX Runtime GPU) supplements face detection to sustain tracking when face is not visible (e.g. turned sideways). Always on when model file exists, graceful fallback if missing.
- **Visit tracking** — Per-location visits with flip-flop prevention, automatic timeout, and transition detection
- **Footage recording** — Continuous VP8/WebM recording from visit start to end at native camera resolution
- **Action detection** — CLIP ViT-B/32 zero-shot classification (e.g., "Using phone", "Typing", "Idle"). Only runs when face detector confirms the person (not on head-only or tracker-only boxes). Optional, toggled via `ACTION_DETECTION_ENABLED` env var.
- **Auto-capture unknowns** — Automatically saves face crops of unrecognised people as `unknown_1`, `unknown_2`, etc. Re-identifies them on reappearance. Best for small crowds. Optional, toggled via `AUTO_CAPTURE_ENABLED` env var.
- **AI kill switch** — Set `FACE_DETECTION_ENABLED=false` to disable all inference (detection, recognition, tracking, attendance) and run as a pure camera stream viewer. Useful for diagnosing lag or running on non-GPU hardware.
- **People management** — Dedicated `/people` page to view all enrolled persons, rename anyone (especially auto-captured unknowns), delete persons, view face images, transfer images between persons, and delete individual images.
- **History dashboard** — Daily summary, per-person, and per-location visit history with footage playback
- **SSE attendance stream** — Real-time attendance events via Server-Sent Events

## Requirements

- **Python 3.11.9**
- **Ubuntu** (for GPU branch)
- **NVIDIA GPU** with CUDA 13.1+ (for GPU branch — tested on RTX 4060 Laptop 8GB)
- A working webcam or RTSP camera source
- `opencv-contrib-python` (not plain `opencv-python`) — needed for CSRT/KCF trackers

## Setup

### 1) Clone and checkout

```bash
git clone <YOUR_REPO_URL>
cd face_recognition
git checkout ubuntu-dev   # for GPU branch
```

### 2) Create virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python --version   # should print Python 3.11.9
```

### 3) Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** If using a tmpfs `/tmp` with limited space, set `TMPDIR` for large installs:
> ```bash
> TMPDIR=/home/main/tmp pip install -r requirements.txt
> ```

> **Important:** `onnxruntime-gpu` and `onnxruntime` (CPU) conflict. If both end up installed, the CPU version overrides GPU. Fix with:
> ```bash
> pip uninstall -y onnxruntime
> ```

#### OpenCV tracker errors

If you see "OpenCV tracker not available" or boxes/labels never appear in the
UI even though detection logs show faces, only `opencv-contrib-python` should
be installed. Both `opencv-python` and `opencv-python-headless` shadow the
contrib build and silently remove the CSRT/KCF trackers — and
`pip install insightface` pulls in `opencv-python-headless` transitively.

```bash
pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
pip install opencv-contrib-python
```

Verify with:

```bash
python -c "import cv2; print(hasattr(cv2, 'TrackerCSRT_create'))"   # must print True
```

### 4) Configure environment

```bash
cp .env.example .env
# Edit .env as needed
```

Key settings in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_PATH` | `face_recognition.db` | SQLite database file path |
| `DATABASE_URL` | *(none)* | PostgreSQL connection string (overrides SQLite) |
| `VISIT_TIMEOUT_MINUTES` | `10` | Close a visit after this many minutes unseen |
| `VISIT_TRANSITION_SECS` | `2.0` | Grace period before transitioning to a new camera |
| `FACE_DETECTION_ENABLED` | `true` | Master AI switch. Set to `false` to disable all face detection, recognition, tracking, and attendance — the system becomes a plain camera stream viewer with no GPU inference load. |
| `ACTION_DETECTION_ENABLED` | `false` | Enable/disable CLIP action detection |
| `AUTO_CAPTURE_ENABLED` | `false` | Enable/disable auto-capture of unknown persons (best for small crowds) |
| `USE_NVDEC` | `true` | Use GPU-side video decode (NVDEC) for RTSP streams. Set to `false` to force CPU decode (debugging / non-NVIDIA hosts). |
| `LIVE_ANNOTATIONS_ENABLED` | `true` | Draw bounding boxes + name labels on the live MJPEG feed. Set to `false` for a clean live stream when scenes get crowded — boxes still appear on saved footage regardless. |
| `FOOTAGE_DIR` | *(required)* | Directory where footage WebM files are written. Set to a network mount path (e.g. `/mnt/camera_system/footage`) to offload storage to a NAS. |

### 5) Add face images

Place reference images in `faces/<person_name>/`:

```
faces/
  john_doe/
    photo1.jpg
    photo2.jpg
  jane_smith/
    img.png
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.webp`

You can also take or upload images via the web UI, or let auto-capture create entries for unknown persons (when enabled).

### 6) Run

```bash
python app.py
```

Open http://127.0.0.1:5000

> The engine does **not** auto-start. Press **Start** in the UI.
> Multi-camera grid mode is the default if a saved grid config exists.

## Project Structure

```
app.py                  # Flask server, API endpoints, visit management
face_engine.py          # Face detection/tracking engine, grid mode, footage
head_detector.py        # YOLOv8n head detector (ONNX Runtime GPU)
action_detector.py      # CLIP zero-shot action classifier (ONNX Runtime GPU)
db.py                   # Database layer (SQLite / PostgreSQL)
templates/
  index.html            # Main dashboard UI (live feed, camera controls)
  history.html          # Visit history dashboard
  people.html           # People management (view, rename, delete, transfer)
faces/                  # Enrolled face images (per-person subdirectories)
footage/                # Recorded visit footage (WebM) — or a NAS mount path via FOOTAGE_DIR
models/
  head-yolov8n-onnx/    # YOLOv8n head detector ONNX model (~12MB)
  clip-vit-base-patch32-onnx/  # CLIP ViT-B/32 ONNX model (~600MB)
grid_config.json        # Saved grid layout + camera slot assignments
hw_capture.py           # NVDEC-accelerated RTSP capture wrapper (cv2.VideoCapture-compatible)
docker-compose.yml      # Optional PostgreSQL via Docker (not required for SQLite)
.env                    # Environment configuration (not committed)
.env.example            # Example environment file
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/video` | GET | MJPEG video stream (503 when stopped) |
| `/api/status` | GET | Engine status, identity count, action detection flag |
| `/api/camera` | GET/POST | Get or set camera source |
| `/api/attendance/stream` | GET | SSE attendance event stream |
| `/api/grid/config` | GET/POST | Get or set grid layout and camera assignments |
| `/api/history/daily` | GET | Daily visit summary |
| `/api/history/person/<name>` | GET | Visit history for a person |
| `/api/history/location/<id>` | GET | Visit history for a location |
| `/api/history/clear` | POST | Clear all visit/session data |
| `/api/people` | GET | List all enrolled persons with thumbnail URLs |
| `/api/rename_person` | POST | Rename a person (moves folder + updates DB) |
| `/api/person/<name>` | DELETE | Delete a person (removes folder + visits) |
| `/api/person/<name>/images` | GET | List all face images for a person |
| `/api/person/<name>/image/<file>` | DELETE | Delete a single face image |
| `/api/person/<name>/image/<file>/transfer` | POST | Move an image to another person |
| `/api/upload_face` | POST | Upload a face image for a person |
| `/history` | GET | History dashboard page |
| `/people` | GET | People management page |

## Tuning

Key parameters in `app.py` engine initialization:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `detect_every` | `0.3` | Seconds between face detection runs |
| `detect_scale` | `1.0` | Scale factor for detection (lower = faster, less accurate) |
| `width` / `height` | `1280` / `720` | Camera resolution |
| `out_fps` | `15` | Target FPS for MJPEG stream and footage |
| `threshold` | `0.5` | Cosine distance threshold for ArcFace face matching. Below this distance → recognised as that person. Raise (toward 0.6) if same person is being mis-labeled "unknown"; lower (toward 0.4) if different people are being collapsed together. |
| `tracker_type` | `CSRT` | OpenCV tracker (`CSRT` or `KCF`) |

## Auto-Capture Unknowns

When `AUTO_CAPTURE_ENABLED=true`:

1. An unrecognised face is tracked. The engine keeps the largest crop seen during the tracking window as the "best frame".
2. After **5 seconds** of continuous tracking (`_unknown_capture_min_seconds`), the best frame is gated through a quality check — InsightFace must re-detect a face in that crop with `det_score ≥ 0.50` and dimensions `≥ 60×60 px`. Crops that fail (motion blur, profile shots, tracker drift onto hands/shoulders) are silently rejected and we wait for a better frame on the next track.
3. On pass: the crop is saved to `faces/unknown_N/1.jpg` and the track is promoted to `unknown_N`.
4. **Bootstrap sample accumulation:** for the first 10 frames after enrolment, additional samples are saved with only a 5-second per-camera cooldown (instead of the steady-state 10 minutes). This builds a diverse averaged template fast, fixing the "single-frame template never matches re-appearances" failure mode.
5. After the bootstrap phase the per-camera cooldown reverts to 10 minutes; a different camera's view of the same person is always allowed immediately. Maximum 30 images per person, 150 auto-captured `unknown_N` folders total.
6. Tracks that lose recognition for 12 consecutive frames (~3.6 s at default cadence) are demoted from `unknown_N` back to "unknown" and become eligible for new enrolment. This patience window is what prevents one person from generating dozens of duplicate folders.
7. Use the **People** page (`/people`) to rename `unknown_N` entries to real names, or delete false captures.

Designed for **small, controlled environments** (offices, labs). Disable it for large public spaces where many strangers would quickly fill the 150-slot cap.

## Maintenance

### Clear all visit history

The "Clear Visit Data" button is hidden in the UI to prevent accidental wipes. To wipe all recorded visits and sessions (keeps people, locations, and face images), run one of the following with the Flask server **stopped**:

```bash
# Option 1 — direct SQLite (fastest, no Python needed):
sqlite3 face_recognition.db "DELETE FROM visits; DELETE FROM sessions; VACUUM;"

# Option 2 — via the Python helper (same code path the API used):
python3 -c "import db; print(db.clear_all_data(), 'visits deleted')"
```

If the server is running, you can also call the API directly:

```bash
curl -X POST http://localhost:5000/api/history/clear
```

## Troubleshooting

**Webcam won't open** — Close other apps using the camera. Try a different `cam_index`.

**"OpenCV tracker not available"** — Install `opencv-contrib-python` (see setup).

**Action detection not loading** — Ensure `onnxruntime-gpu` is installed and `onnxruntime` (CPU) is not. Check CUDA drivers with `nvidia-smi`.

**Footage won't play in browser** — Files are VP8/WebM. All modern browsers support this. The OpenCV "tag VP80 is not supported" warning is misleading — the files are valid.

**Camera lag on repeated start/stop** — Fixed in recent updates. The engine now fully cleans up GPU models, footage writers, frame buffers, and pending state on stop.

**No bounding boxes / labels appear in UI even though detection logs show faces** — Almost always means the OpenCV CSRT tracker isn't available. Run the OpenCV cleanup in the setup section. `pip install insightface` pulls in `opencv-python-headless` which silently shadows `opencv-contrib-python`.

**Stale `.embeddings.npz` files** — Legacy DeepFace/Facenet512 caches. The current code uses `.arcface.npz` (InsightFace). On first run the new caches regenerate automatically; the old files are unused but can be deleted with `find faces -name ".embeddings.npz" -delete`.

**Recognition threshold tuning** — InsightFace's ArcFace embeddings are more discriminative than the previous Facenet512. If you see too many "unknown" labels for known faces, raise `threshold` in `app.py` (e.g., 0.4 → 0.5). If you see false matches between different people, lower it (0.4 → 0.35).

**NVDEC not engaging** — Check `nvidia-smi dmon -s u` while cameras are running; the `dec` column should be > 0 if NVDEC is active. If it stays at 0, the hardware decode init is silently falling back to CPU. Common causes: codec the GPU doesn't support (older Pascal-and-earlier cards lack HEVC 10-bit / AV1 hardware decode), or stale FFmpeg shared libraries the PyAV wheel can't find. Force-disable NVDEC with `USE_NVDEC=0` in `.env` to confirm it's the culprit.
