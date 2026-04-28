
# Face Recognition Attendance System

A Flask-based face recognition and attendance system with multi-camera support, visit tracking, footage recording, CLIP-based action detection, YOLO head detection, and auto-capture of unknown persons.

Uses DeepFace for face recognition, OpenCV for tracking, CLIP (ONNX Runtime) for zero-shot action classification, and YOLOv8n for head detection.

## Branches

| Branch | Target | Notes |
|--------|--------|-------|
| `main` | CPU (Windows / Ubuntu) | No GPU acceleration |
| `ubuntu-dev` | **GPU (Ubuntu + NVIDIA)** | CUDA + ONNX Runtime GPU. **Do not merge into main.** |

## Features

- **Multi-camera grid** — 2x2, 3x3, or 4x4 camera grid with per-tile face detection and tracking
- **Face recognition** — DeepFace (RetinaFace detector + Facenet512 embeddings) with CSRT tracking
- **Head detection** — YOLOv8n (ONNX Runtime GPU) supplements face detection to sustain tracking when face is not visible (e.g. turned sideways). Always on when model file exists, graceful fallback if missing.
- **Visit tracking** — Per-location visits with flip-flop prevention, automatic timeout, and transition detection
- **Footage recording** — Continuous VP8/WebM recording from visit start to end at native camera resolution
- **Action detection** — CLIP ViT-B/32 zero-shot classification (e.g., "Using phone", "Typing", "Idle"). Only runs when face detector confirms the person (not on head-only or tracker-only boxes). Optional, toggled via `ACTION_DETECTION_ENABLED` env var.
- **Auto-capture unknowns** — Automatically saves face crops of unrecognised people as `unknown_1`, `unknown_2`, etc. Re-identifies them on reappearance. Best for small crowds. Optional, toggled via `AUTO_CAPTURE_ENABLED` env var.
- **People management** — Dedicated `/people` page to view all enrolled persons, rename anyone (especially auto-captured unknowns), delete persons, view face images, transfer images between persons, and delete individual images.
- **History dashboard** — Daily summary, per-person, and per-location visit history with footage playback
- **Screenshot capture** — Automatic per-visit screenshot on first detection
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

If you see "OpenCV tracker not available":

```bash
pip uninstall -y opencv-python
pip install opencv-contrib-python
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
| `ACTION_DETECTION_ENABLED` | `false` | Enable/disable CLIP action detection |
| `AUTO_CAPTURE_ENABLED` | `false` | Enable/disable auto-capture of unknown persons (best for small crowds) |

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
footage/                # Recorded visit footage (WebM)
screenshots/            # Visit screenshots (JPEG)
models/
  head-yolov8n-onnx/    # YOLOv8n head detector ONNX model (~12MB)
  clip-vit-base-patch32-onnx/  # CLIP ViT-B/32 ONNX model (~600MB)
grid_config.json        # Saved grid layout + camera slot assignments
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
| `threshold` | `0.4` | Cosine distance threshold for face matching |
| `tracker_type` | `CSRT` | OpenCV tracker (`CSRT` or `KCF`) |

## Auto-Capture Unknowns

When `AUTO_CAPTURE_ENABLED=true`:

1. The engine tracks unrecognised faces and accumulates consecutive face-detector-confirmed detections
2. After 4 consecutive detections (~1.2 seconds), it saves a cropped face image to `faces/unknown_N/1.jpg`
3. The face is immediately enrolled — on reappearance, it's recognised as `unknown_N` (not generic "unknown")
4. Maximum 50 auto-captured unknowns (configurable in `face_engine.py`)
5. Use the **People** page (`/people`) to rename `unknown_N` entries to real names, or delete false captures

This feature is designed for **small, controlled environments** (offices, labs). Disable it for large public spaces where dozens of strangers would quickly fill the 50-slot cap.

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
