
# Face Recognition Attendance System

A Flask-based face recognition and attendance system with multi-camera support, visit tracking, footage recording, and CLIP-based action detection.

Uses DeepFace for face recognition, OpenCV for tracking, and CLIP (ONNX Runtime) for zero-shot action classification.

## Branches

| Branch | Target | Notes |
|--------|--------|-------|
| `main` | CPU (Windows / Ubuntu) | No GPU acceleration |
| `ubuntu-dev` | **GPU (Ubuntu + NVIDIA)** | CUDA + ONNX Runtime GPU. **Do not merge into main.** |

## Features

- **Multi-camera grid** — 2x2, 3x3, or 4x4 camera grid with per-tile face detection and tracking
- **Face recognition** — DeepFace (RetinaFace detector + Facenet512 embeddings) with CSRT tracking
- **Visit tracking** — Per-location visits with flip-flop prevention, automatic timeout, and transition detection
- **Footage recording** — Continuous VP8/WebM recording from visit start to end at native camera resolution
- **Action detection** — CLIP ViT-B/32 zero-shot classification (e.g., "Using phone", "Typing", "Idle"). Optional, toggled via env var.
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
| `ACTION_DETECTION_ENABLED` | `true` | Enable/disable CLIP action detection |

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

You can also upload images via the web UI.

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
action_detector.py      # CLIP zero-shot action classifier (ONNX Runtime GPU)
db.py                   # Database layer (SQLite / PostgreSQL)
templates/
  index.html            # Main dashboard UI (live feed, camera controls)
  history.html          # Visit history dashboard
faces/                  # Enrolled face images (per-person subdirectories)
footage/                # Recorded visit footage (WebM)
screenshots/            # Visit screenshots (JPEG)
models/                 # Cached ONNX models (auto-downloaded on first run)
grid_config.json        # Saved grid layout + camera slot assignments
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
| `/history` | GET | History dashboard page |

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

## Troubleshooting

**Webcam won't open** — Close other apps using the camera. Try a different `cam_index`.

**"OpenCV tracker not available"** — Install `opencv-contrib-python` (see setup).

**Action detection not loading** — Ensure `onnxruntime-gpu` is installed and `onnxruntime` (CPU) is not. Check CUDA drivers with `nvidia-smi`.

**Footage won't play in browser** — Files are VP8/WebM. All modern browsers support this. The OpenCV "tag VP80 is not supported" warning is misleading — the files are valid.

**Single-cam mode is laggy on builtin webcam** — Known issue with camera 0 at 1280x720 (~10 FPS). Use multi-cam grid mode as a workaround.
