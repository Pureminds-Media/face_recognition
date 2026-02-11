
# Face Recognition Dashboard (Flask + Webcam)

A small Flask dashboard that streams a webcam feed (MJPEG) and does face recognition / tracking using DeepFace + OpenCV. Images live under `faces/<name>/` and can be uploaded via the web UI. The app also supports “attendance” marking over SSE.

## <span style="color:red">**IMPORTANT!!!**</span>
There are two branches: `main` and `ubuntu-dev`
- `main` — Uses CPU for processing. Runs on both Windows and Ubuntu
- `ubuntu-dev` — Uses GPU for processing. ONLY runs on Ubuntu

`main` and `ubuntu-dev` should <span style="color:red">**NEVER**</span> be merged into each other since they are for different setups.

## Repo Structure

- `app.py` — Flask server + API endpoints + MJPEG `/video`
- `face_engine.py` — webcam capture, detection/embedding, tracking, JPEG output
- `templates/index.html` — dashboard UI

Images:
- `faces/<person>/1.jpg`, `2.jpg`, …

## Requirements

- **Python 3.11.9** (venv uses this)
- A working webcam
- OS packages needed for OpenCV/webcam access (depends on OS)

> **Important:** `face_engine.py` uses OpenCV trackers (CSRT/KCF). These require **opencv-contrib-python** (not plain opencv-python). If you see “OpenCV tracker not available”, install contrib.

## Setup (Windows / macOS / Linux)

### 1) Clone the repo
```bash
git clone <YOUR_REPO_URL>
cd <YOUR_REPO_FOLDER>
```

### 2) Create a Python 3.11.9 virtual environment

**Windows (PowerShell)**
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

**macOS / Linux**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python --version
```

Confirm it prints **Python 3.11.9**.

### 3) Install dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### If you hit OpenCV tracker errors
Install contrib build (and remove plain opencv-python if both are installed):

```bash
pip uninstall -y opencv-python
pip install opencv-contrib-python
```

### 4) Run the app
```bash
python app.py
```

Open:
- http://127.0.0.1:5000

> **Note:** Don’t run Flask with the reloader for webcam apps (it starts twice). This repo already disables it in `app.py` (`use_reloader=False`).

## Usage

### Start / Stop Camera
Use the Start/Stop buttons in the UI.

Video stream endpoint:
- `GET /video` — MJPEG stream
  - Returns **503** when stopped.

### Add Images
Use “Add image” in the UI:
- Choose existing name or type a new name
- Upload an image
- Saved as `faces/<person>/<N>.<ext>`

### Reload Faces
Use “Reload faces” to rebuild embeddings from disk.

## Attendance Mode (Biometric Attendance)

The UI listens via Server-Sent Events (SSE) and shows attendance in real time:

- `GET /api/attendance/stream` — SSE stream:
  - `event: state` contains `{ running, attendance }`
  - `event: new` when someone is marked first time
  - `event: repeat` when they re-appear after disappearing (“already attended”)

Attendance is currently **in-memory** (resets when the server restarts).

## Tuning / Performance

Recognition quality at distance is affected by face size and embedding quality.

In `face_engine.py`, the detector filters small faces:
- `if w < 90 or h < 90: continue`  
Lower this value (e.g., 60) to allow recognizing smaller/farther faces (may increase false detections).

Match threshold:
- `threshold` controls cosine distance cutoff; higher allows more matches (more false positives possible).

If CPU is high:
- Reduce `width/height`, `out_fps`, or increase `detect_every`.
- Consider using `tracker_type="KCF"` (lighter than CSRT).

## Troubleshooting

### Webcam won’t open
- Close other apps using the camera (Zoom, Teams, etc.)
- If `cv2.VideoCapture(0)` fails, try changing `cam_index` in `FaceEngine`.

### “OpenCV tracker not available”
Install `opencv-contrib-python` (see above).

### UI shows broken image on refresh (503)
Make sure `templates/index.html` does **not** set the `<img>` src to `/video` initially; it should start empty and only connect when running.