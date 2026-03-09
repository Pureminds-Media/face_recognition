import os
import atexit
import time
import re
import json
import glob
import uuid
import logging
import mimetypes
import subprocess
from datetime import datetime, date, timezone, timedelta
from queue import Queue, Empty, Full
from flask import Flask, Response, render_template, request, jsonify, send_from_directory, stream_with_context
from werkzeug.utils import secure_filename
import threading
import cv2
from dotenv import load_dotenv
from face_engine import FaceEngine, AVAILABLE_LAYOUTS
import db

load_dotenv()
log = logging.getLogger(__name__)

app = Flask(__name__)

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
FACES_DIR = "faces"
TEST_UPLOAD_DIR = os.path.join("test_runs", "uploads")
TEST_OUTPUT_DIR = os.path.join("test_runs", "outputs")
SCREENSHOTS_DIR = "screenshots"
FOOTAGE_DIR = "footage"
ALLOWED_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
os.makedirs(FOOTAGE_DIR, exist_ok=True)

engine = FaceEngine(
    known_dir=FACES_DIR,
    detector="retinaface",
    model="Facenet512",
    threshold=0.4,
    detect_every=0.3,
    detect_scale=1.0,
    tracker_type="CSRT",
    width=1280,
    height=720,
    out_fps=15,
    jpeg_quality=80,
)

# Load saved grid config (if any) and default to multi-camera mode
_saved_grid = FaceEngine.load_grid_config()
if _saved_grid is not None:
    try:
        engine.set_grid_layout(*_saved_grid["layout"])
        engine.cam_index = _saved_grid["cam_index"]
    except (ValueError, KeyError):
        pass  # invalid layout/config, keep default

# Engine stays stopped until user clicks Start in the UI
# engine.start()
# To allow starting/stopping of camera
state_lock = threading.Lock()

# Attendance state (in-memory, per server run)
attendance_lock = threading.Lock()
attendance_state = {}  # name -> {attended: bool, first_seen_ts: float, last_seen_mono: float, present: bool}
attendance_events_q = Queue(maxsize=200)
ATTENDANCE_LOOP_SECS = 0.3
ATTENDANCE_DISAPPEAR_SECS = 2.0
UNKNOWN_PROMPT_SECS = 10.0
UNKNOWN_RESET_GRACE_SECS = 1.5
VISIT_TIMEOUT_MINUTES = int(os.getenv("VISIT_TIMEOUT_MINUTES", "10"))
VISIT_STALE_CHECK_SECS = 30.0  # how often to check for stale visits
VISIT_TRANSITION_SECS = float(os.getenv("VISIT_TRANSITION_SECS", "2.0"))  # seconds absent from current camera before transitioning to a new location
qr_prompt_state = {
    "unknown_first_mono": None,
    "unknown_last_seen_mono": None,
    "active": False,
}
test_jobs_lock = threading.Lock()
test_job_run_lock = threading.Lock()
test_jobs = {}  # job_id -> {status, progress, error, result_url, ...}

# --- Database init ---
db.init_db()
_current_session_id = None
if db.is_available():
    _current_session_id = db.create_session(
        camera_source=str(engine.cam_index) if engine.cam_index is not None else None
    )
    # Sync locations from grid config
    _saved_grid = FaceEngine.load_grid_config()
    if _saved_grid and _saved_grid.get("slots"):
        for _src, _name in FaceEngine.get_slot_locations(_saved_grid["slots"]).items():
            db.upsert_location(_src, _name)
    # Ensure single-camera source has a location too
    if not engine._is_grid_mode():
        _src = str(engine.cam_index)
        db.upsert_location(_src, f"Camera {_src}")

    log.info("DB session started: %s", _current_session_id)

# In-memory visit tracking state: person_name -> {location_id, visit_id, camera_source}
_active_visits = {}
_last_stale_check = time.monotonic()


def _shutdown_db():
    footage_results = engine.stop_all_footage()
    if db.is_available():
        # Save visible_duration for all visits that had active footage writers
        for vid, (fname, visible_secs) in footage_results.items():
            if visible_secs > 0:
                try:
                    db.update_visit_visible_duration(vid, visible_secs)
                except Exception:
                    pass
        db.close_all_open_visits()
        db.end_session(_current_session_id)
        db.close_db()

atexit.register(_shutdown_db)


def _q_put_latest(q, item):
    try:
        q.put_nowait(item)
    except Full:
        try:
            q.get_nowait()
        except Empty:
            pass
        q.put_nowait(item)

def _set_test_job(job_id: str, **fields):
    with test_jobs_lock:
        job = dict(test_jobs.get(job_id, {}))
        job.update(fields)
        job["updated_ts"] = time.time()
        test_jobs[job_id] = job

def _run_test_job(job_id: str, input_path: str):
    out_path = os.path.join(TEST_OUTPUT_DIR, f"{job_id}.mp4")

    def _progress(done: int, total: int):
        progress = 0.0
        if total and total > 0:
            progress = min(100.0, round((float(done) / float(total)) * 100.0, 1))
        _set_test_job(
            job_id,
            status="processing",
            progress=progress,
            frame_index=int(done),
            total_frames=int(total) if total else None,
        )

    try:
        _set_test_job(job_id, status="processing", progress=0.0)
        os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

        with test_job_run_lock:
            engine.reload_faces()
            out_info = engine.process_video_file(input_path, out_path, progress_cb=_progress)
            final_output_path = out_info.get("output_path", out_path)
            final_mime = out_info.get("mime") or mimetypes.guess_type(final_output_path)[0] or "application/octet-stream"
            out_filename = os.path.basename(final_output_path)

        _set_test_job(
            job_id,
            status="done",
            progress=100.0,
            result_url=f"/test/results/{out_filename}",
            result_mime=final_mime,
        )
    except Exception as e:
        _set_test_job(job_id, status="error", error=str(e))

def _attendance_roster():
    # include all known identities, even if not attended yet
    names = set()

    # from embeddings (loaded faces)
    for name, _ in getattr(engine, "known_embeddings", []):
        names.add(name)

    # plus anyone already seen (safety)
    names.update(attendance_state.keys())

    roster = []
    for name in sorted(names):
        s = attendance_state.get(name)
        roster.append(
            {
                "name": name,
                "attended": bool(s.get("attended", False)) if s else False,
                "present": bool(s.get("present", False)) if s else False,
                "first_seen_ts": float(s.get("first_seen_ts", 0.0)) if s else 0.0,
            }
        )
    return roster

def _attendance_loop():
    last_sig = None
    last_running = None

    while True:
        running = engine.is_running()
        events = []
        tracks = []

        with attendance_lock:
            if running:
                tracks = engine.get_tracks()
                events = _update_attendance_from_tracks(tracks)
            else:
                if last_running:
                    _mark_all_absent()

            _update_qr_prompt_from_tracks(tracks, running)
            roster = _attendance_roster()
            prompt_payload = _get_qr_prompt_payload()
            sig = (
                running,
                prompt_payload["qr_prompt"],
                tuple((r["name"], r["attended"], r["present"]) for r in roster),
            )

        # Emit state only if it changed OR if we have events
        if sig != last_sig or events:
            _q_put_latest(
                attendance_events_q,
                {
                    "event": "state",
                    "data": {
                        "running": running,
                        "attendance": roster,
                        "qr_prompt": prompt_payload["qr_prompt"],
                        "unknown_elapsed_secs": prompt_payload["unknown_elapsed_secs"],
                    },
                },
            )
            for ev in events:
                _q_put_latest(attendance_events_q, {"event": ev["type"], "data": ev})
            last_sig = sig

        last_running = running
        time.sleep(ATTENDANCE_LOOP_SECS if running else 1.0)


def _update_attendance_from_tracks(tracks):
    """Update attendance_state and DB visits based on current recognized tracks.

    - First time a person is seen: mark attended once and emit a 'new' event
    - If person disappears (not seen for ATTENDANCE_DISAPPEAR_SECS) and later reappears: emit a 'repeat' event
    - DB: open/update/close visits per person per location
    """
    global _last_stale_check
    now_mono = time.monotonic()
    now_ts = time.time()
    events = []

    for t in (tracks or []):
        name = (t or {}).get("name")
        if not name or name == "unknown":
            continue

        # Determine camera source for this track
        camera_source = str((t or {}).get("camera_source", engine.cam_index))
        confidence = (t or {}).get("best")
        last_detect_t = float((t or {}).get("last_detect_t", 0.0))

        s = attendance_state.get(name)
        if s is None:
            attendance_state[name] = {
                "attended": True,
                "first_seen_ts": now_ts,
                "last_seen_mono": now_mono,
                "present": True,
            }
            events.append({"type": "new", "name": name})
        else:
            # re-appearance -> toast "already attended"
            if not s.get("present", False):
                s["present"] = True
                events.append({"type": "repeat", "name": name})
            s["last_seen_mono"] = now_mono

        # --- DB visit tracking ---
        if db.is_available():
            _update_visit_for_person(name, camera_source, confidence, last_detect_t)

    # Mark disappeared after a grace period (avoid flicker)
    for name, s in list(attendance_state.items()):
        if s.get("present", False) and (now_mono - s.get("last_seen_mono", now_mono)) > ATTENDANCE_DISAPPEAR_SECS:
            s["present"] = False

    # Periodically close stale visits in DB
    if db.is_available() and (now_mono - _last_stale_check) >= VISIT_STALE_CHECK_SECS:
        _last_stale_check = now_mono
        closed = db.close_stale_visits(VISIT_TIMEOUT_MINUTES)
        if closed:
            # Remove from in-memory tracking and close footage writers
            for name in list(_active_visits.keys()):
                v = _active_visits[name]
                if v.get("visit_id"):
                    # Check if this visit was closed
                    open_v = db.get_open_visit(name, v["location_id"])
                    if open_v is None:
                        _stop_visit_footage(v["visit_id"])
                        del _active_visits[name]

    return events


def _save_visit_screenshot(visit_id, person_name, camera_source):
    """Grab the buffered snapshot from the engine and save it for a visit."""
    try:
        snap = engine.get_snapshot(person_name, camera_source)
        if snap is None:
            return
        fname = f"visit_{visit_id}.jpg"
        fpath = os.path.join(SCREENSHOTS_DIR, fname)
        with open(fpath, "wb") as f:
            f.write(snap)
        db.update_visit_screenshot(visit_id, fname)
    except Exception as e:
        log.debug("Failed to save visit screenshot: %s", e)


def _start_visit_footage(visit_id, person_name, camera_source):
    """Open a streaming VideoWriter for a visit.

    The writer receives frames from the engine's render loop only while the
    person is visible on the camera.
    """
    try:
        fname = f"visit_{visit_id}.webm"
        ok, actual_fname = engine.start_footage(
            visit_id, person_name, camera_source, FOOTAGE_DIR, fname
        )
        if ok:
            db.update_visit_footage(visit_id, actual_fname)
            log.debug("Started footage writer for visit %s -> %s", visit_id, actual_fname)
        else:
            log.debug("Failed to open footage writer for visit %s", visit_id)
    except Exception as e:
        log.debug("Failed to start footage recording: %s", e)


def _stop_visit_footage(visit_id):
    """Close the footage VideoWriter for a visit and save visible_duration."""
    try:
        fname, visible_secs = engine.stop_footage(visit_id)
        if fname:
            log.debug("Closed footage writer for visit %s -> %s (%.1fs visible)",
                       visit_id, fname, visible_secs)
            if visible_secs > 0:
                db.update_visit_visible_duration(visit_id, visible_secs)
    except Exception as e:
        log.debug("Failed to stop footage for visit %s: %s", visit_id, e)


def _update_visit_for_person(name, camera_source, confidence=None, last_detect_t=0.0):
    """Open, update, or transition a visit for a person at a camera/location.

    Flip-flop prevention: when a person appears on a *different* camera than
    their current active visit, we do NOT immediately transition.  Instead we
    check how recently they were seen on the *original* camera:

      - If within VISIT_TRANSITION_SECS: they are likely in an overlap
        zone — keep the existing visit, just refresh its DB timestamp.
      - If older than VISIT_TRANSITION_SECS: they have genuinely left the
        original camera — close old visit, open new one at the new location.

    Ghost-box prevention: only detector-confirmed tracks (fresh last_detect_t)
    refresh last_seen_mono.  Tracker-only ghost boxes still bump DB last_seen
    but do NOT prevent visit transitions.
    """
    now_mono = time.monotonic()
    # A track is "detector-confirmed" if the face detector saw it recently
    # (within 2x detect_every).  Ghost boxes from CSRT have stale last_detect_t.
    detect_freshness = getattr(engine, "detect_every", 1.0) * 2.0
    is_detector_confirmed = (now_mono - last_detect_t) < detect_freshness

    loc = db.get_location_by_source(camera_source)
    if loc is None:
        # Auto-create location for unknown camera sources
        loc_id = db.upsert_location(camera_source, f"Camera {camera_source}")
        loc = {"id": loc_id, "camera_source": camera_source, "name": f"Camera {camera_source}"}
    loc_id = loc["id"]

    active = _active_visits.get(name)

    if active is None:
        # No active visit — open a new one
        vid = db.open_visit(name, loc_id, confidence=confidence, session_id=_current_session_id)
        _active_visits[name] = {
            "location_id": loc_id,
            "visit_id": vid,
            "camera_source": camera_source,
            "last_seen_mono": now_mono,
        }
        _save_visit_screenshot(vid, name, camera_source)
        _start_visit_footage(vid, name, camera_source)
    elif active["location_id"] == loc_id:
        # Same location — update last_seen in DB
        db.update_visit_seen(active["visit_id"], confidence=confidence)
        # Only refresh the monotonic clock if the face detector recently
        # confirmed this track.  Tracker-only ghost boxes must NOT keep
        # the timer alive, or they block visit transitions to other cameras.
        if is_detector_confirmed:
            active["last_seen_mono"] = now_mono
    else:
        # Different location — check whether person has left original camera
        elapsed = now_mono - active.get("last_seen_mono", 0)
        if elapsed < VISIT_TRANSITION_SECS:
            # Still recently seen on original camera (overlap zone) — do NOT
            # transition; just keep the existing visit alive.
            db.update_visit_seen(active["visit_id"], confidence=confidence)
            log.debug("Visit overlap: %s seen on cam %s but active on loc %s (%.1fs ago, need %.1fs)",
                       name, camera_source, active["location_id"], elapsed, VISIT_TRANSITION_SECS)
        else:
            # Person has been absent from original camera long enough —
            # genuine transition: close old visit, open new one.
            log.info("Visit transition: %s from loc %s -> cam %s (absent %.1fs >= %.1fs)",
                      name, active["location_id"], camera_source, elapsed, VISIT_TRANSITION_SECS)
            _stop_visit_footage(active["visit_id"])
            db.close_visit(active["visit_id"])
            vid = db.open_visit(name, loc_id, confidence=confidence, session_id=_current_session_id)
            _active_visits[name] = {
                "location_id": loc_id,
                "visit_id": vid,
                "camera_source": camera_source,
                "last_seen_mono": now_mono,
            }
            _save_visit_screenshot(vid, name, camera_source)
            _start_visit_footage(vid, name, camera_source)

def _mark_all_absent():
    now_mono = time.monotonic()
    for s in attendance_state.values():
        s["present"] = False
        s["last_seen_mono"] = now_mono
    qr_prompt_state["unknown_first_mono"] = None
    qr_prompt_state["unknown_last_seen_mono"] = None
    qr_prompt_state["active"] = False


def _update_qr_prompt_from_tracks(tracks, running: bool):
    now_mono = time.monotonic()
    has_unknown = any((t or {}).get("name") == "unknown" for t in (tracks or []))

    if not running:
        qr_prompt_state["unknown_first_mono"] = None
        qr_prompt_state["unknown_last_seen_mono"] = None
        qr_prompt_state["active"] = False
        return

    if has_unknown:
        if qr_prompt_state["unknown_first_mono"] is None:
            qr_prompt_state["unknown_first_mono"] = now_mono
        qr_prompt_state["unknown_last_seen_mono"] = now_mono
    else:
        last_seen = qr_prompt_state.get("unknown_last_seen_mono")
        if (last_seen is None) or ((now_mono - last_seen) > UNKNOWN_RESET_GRACE_SECS):
            qr_prompt_state["unknown_first_mono"] = None
            qr_prompt_state["unknown_last_seen_mono"] = None
            qr_prompt_state["active"] = False
            return

    first_seen = qr_prompt_state.get("unknown_first_mono")
    qr_prompt_state["active"] = bool(
        first_seen is not None and (now_mono - first_seen) >= UNKNOWN_PROMPT_SECS
    )


def _get_qr_prompt_payload():
    now_mono = time.monotonic()
    first_seen = qr_prompt_state.get("unknown_first_mono")
    unknown_elapsed_secs = 0.0
    if first_seen is not None:
        unknown_elapsed_secs = max(0.0, now_mono - first_seen)

    return {
        "qr_prompt": bool(qr_prompt_state.get("active", False)),
        "unknown_elapsed_secs": round(unknown_elapsed_secs, 1),
    }

def _parse_qr_to_name(raw: str) -> str:
    """
    Accepts:
      - JSON: {"name":"Ahmed_AlQahtani"}
      - prefix: name:Ahmed_AlQahtani
      - plain: Ahmed_AlQahtani
    Returns sanitized person name or "".
    """
    raw = (raw or "").strip()
    if not raw:
        return ""

    # JSON payload
    if raw.startswith("{") and raw.endswith("}"):
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and "name" in obj:
                return safe_person_name(str(obj.get("name", "")))
        except Exception:
            pass

    # prefix payload
    if raw.lower().startswith("name:"):
        return safe_person_name(raw.split(":", 1)[1].strip())

    # fallback: treat as direct name
    return safe_person_name(raw)

def _allowed_video_ext(filename: str) -> str:
    ext = os.path.splitext(filename or "")[1].lower()
    return ext if ext in ALLOWED_VIDEO_EXTS else ""


def _mark_attendance_from_qr(name: str, raw: str):
    """Mark attendance and emit the same event types as face recognition."""
    if not name:
        return

    now_mono = time.monotonic()
    now_ts = time.time()
    ev_type = None

    with attendance_lock:
        if not qr_prompt_state.get("active", False):
            return

        s = attendance_state.get(name)
        if s is None:
            attendance_state[name] = {
                "attended": True,
                "first_seen_ts": now_ts,
                "last_seen_mono": now_mono,
                "present": True,
            }
            ev_type = "new"
        else:
            ev_type = "repeat" if s.get("attended", False) else "new"
            s["attended"] = True
            s["present"] = True
            s["last_seen_mono"] = now_mono
        qr_prompt_state["unknown_first_mono"] = None
        qr_prompt_state["unknown_last_seen_mono"] = None
        qr_prompt_state["active"] = False

    if ev_type:
        _q_put_latest(attendance_events_q, {"event": ev_type, "data": {"name": name}})


def _qr_loop():
    """Poll FaceEngine QR state and mark attendance when a fresh QR appears."""
    last_qr_t = 0.0
    while True:
        try:
            raw, qr_t = engine.get_qr_state()
        except Exception:
            raw = None
            qr_t = 0.0

        raw = (raw or "").strip()
        if raw and qr_t > last_qr_t:
            last_qr_t = qr_t
            name = _parse_qr_to_name(raw)
            _mark_attendance_from_qr(name, raw)

        time.sleep(0.1)

threading.Thread(target=_attendance_loop, daemon=True).start()
threading.Thread(target=_qr_loop, daemon=True).start()

def safe_person_name(name: str) -> str:
    """
    Convert user input to a safe folder name.
    Keeps letters/numbers/_- and turns spaces into underscores.
    """
    name = (name or "").strip()
    name = name.replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9_\-]", "", name)
    # avoid empty
    return name[:64] if name else ""


def next_image_filename(person_dir: str, ext: str) -> str:
    """
    Return next numeric filename like 1.jpg, 2.jpg...
    Scans existing files and picks max+1.
    """
    max_id = 0
    if os.path.isdir(person_dir):
        for fn in os.listdir(person_dir):
            base, e = os.path.splitext(fn)
            if e.lower() not in ALLOWED_EXTS:
                continue
            if base.isdigit():
                max_id = max(max_id, int(base))
    return f"{max_id + 1}{ext}"


def list_people():
    os.makedirs(FACES_DIR, exist_ok=True)
    people = []
    for person in sorted(os.listdir(FACES_DIR)):
        person_dir = os.path.join(FACES_DIR, person)
        if not os.path.isdir(person_dir):
            continue

        imgs = [f for f in os.listdir(person_dir) if os.path.splitext(f)[1].lower() in ALLOWED_EXTS]
        imgs.sort(key=lambda x: (int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else 10**9, x))

        thumb = imgs[0] if imgs else None
        people.append(
            {
                "name": person,
                "count": len(imgs),
                "thumbnail_url": f"/faces/{person}/{thumb}" if thumb else None,
            }
        )
    return people


def _camera_source_to_text(source):
    if source is None:
        return ""
    text = str(source)
    if text.startswith("grid:"):
        rows, cols = engine._grid_layout
        return f"grid_{rows}x{cols}"
    return text


def _list_camera_devices():
    labels = {}
    try:
        out = subprocess.check_output(
            ["v4l2-ctl", "--list-devices"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        current_label = ""
        for line in out.splitlines():
            if not line.strip():
                continue
            if line[:1].isspace():
                node = line.strip()
                m = re.fullmatch(r"/dev/video(\d+)", node)
                if m and current_label:
                    labels[int(m.group(1))] = current_label
            else:
                current_label = line.strip().rstrip(":")
    except Exception:
        pass

    entries = []
    nodes = sorted(
        glob.glob("/dev/video[0-9]*"),
        key=lambda p: int(re.search(r"(\d+)$", p).group(1)),
    )
    for path in nodes:
        m = re.search(r"(\d+)$", path)
        if not m:
            continue
        idx = int(m.group(1))
        entries.append(
            {
                "idx": idx,
                "path": path,
                "label": labels.get(idx),
            }
        )

    devices = []
    seen_labels = set()
    for e in entries:
        # Many cameras expose multiple /dev/video* nodes under one label.
        # Keep only the first node for each labeled physical camera.
        if e["label"]:
            if e["label"] in seen_labels:
                continue
            seen_labels.add(e["label"])
            label = e["label"]
        else:
            # Fallback when v4l2 label is unavailable.
            label = f"Camera {e['idx']}"

        devices.append(
            {
                "value": str(e["idx"]),
                "label": f"{label} ({e['path']})",
                "path": e["path"],
            }
        )

    if devices:
        # Insert grid options for each supported layout
        for idx, (r, c) in enumerate(reversed(AVAILABLE_LAYOUTS)):
            devices.insert(
                0,
                {
                    "value": f"grid_{r}x{c}",
                    "label": f"{r}x{c} Grid ({r * c} slots)",
                    "path": "",
                },
            )
    return devices


def mjpeg_generator():
    last_id = None
    interval = 1.0 / max(1, getattr(engine, "out_fps", 15))

    while engine.is_running():
        frame = engine.get_jpeg()
        if frame is None:
            time.sleep(0.05)
            continue

        # don't spam the same frame
        fid = id(frame)
        if fid == last_id:
            time.sleep(0.005)
            continue
        last_id = fid

        try:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame
                + b"\r\n"
            )
        except (GeneratorExit, BrokenPipeError, ConnectionResetError):
            break

        time.sleep(interval)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/test")
def test_page():
    return render_template("test.html")

@app.route("/video")
def video():
    if not engine.is_running():
        return ("Camera stopped", 503)

    return Response(
        mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

@app.route("/api/test/upload", methods=["POST"])
def api_test_upload():
    f = request.files.get("file")
    if not f:
        return jsonify({"ok": False, "error": "file is required"}), 400

    original = secure_filename(f.filename or "")
    ext = _allowed_video_ext(original)
    if not ext:
        return jsonify({"ok": False, "error": "unsupported video format"}), 400

    os.makedirs(TEST_UPLOAD_DIR, exist_ok=True)
    job_id = uuid.uuid4().hex
    in_name = f"{job_id}{ext}"
    in_path = os.path.join(TEST_UPLOAD_DIR, in_name)
    f.save(in_path)

    _set_test_job(
        job_id,
        status="queued",
        progress=0.0,
        filename=original or in_name,
        result_url=None,
        error=None,
        created_ts=time.time(),
    )
    threading.Thread(target=_run_test_job, args=(job_id, in_path), daemon=True).start()
    return jsonify({"ok": True, "job_id": job_id})

@app.route("/api/test/status/<job_id>")
def api_test_status(job_id):
    with test_jobs_lock:
        job = test_jobs.get(job_id)
    if not job:
        return jsonify({"ok": False, "error": "job not found"}), 404
    return jsonify({"ok": True, "job": job})

@app.route("/test/results/<path:filename>")
def test_result_file(filename):
    guessed = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    return send_from_directory(TEST_OUTPUT_DIR, filename, mimetype=guessed)

@app.route("/api/tracks")
def api_tracks():
    return jsonify({"tracks": engine.get_tracks()})

@app.route("/api/attendance")
def api_attendance():
    """Biometric attendance polling endpoint.

    Returns:
      - running: whether camera is running
      - attendance: roster of attendees with present/attended flags
      - events: toast events {type: 'new'|'repeat', name: str}
      - identities: number of known identities loaded
    """
    running = engine.is_running()

    with attendance_lock:
        tracks = engine.get_tracks() if running else []
        events = _update_attendance_from_tracks(tracks) if running else []
        _update_qr_prompt_from_tracks(tracks, running)
        prompt_payload = _get_qr_prompt_payload()
        roster = [
            {
                "name": name,
                "attended": bool(s.get("attended", True)),
                "present": bool(s.get("present", False)),
                "first_seen_ts": float(s.get("first_seen_ts", 0.0)),
            }
            for name, s in attendance_state.items()
        ]
        roster.sort(key=lambda x: x.get("first_seen_ts", 0.0))

    return jsonify(
        {
            "running": running,
            "attendance": roster,
            "events": events,
            "identities": len(engine.known_embeddings),
            "qr_prompt": prompt_payload["qr_prompt"],
            "unknown_elapsed_secs": prompt_payload["unknown_elapsed_secs"],
        }
    )

@app.route("/api/attendance/reset", methods=["POST"])
def api_attendance_reset():
    with attendance_lock:
        attendance_state.clear()
    return jsonify({"ok": True})

@app.route("/api/people")
def api_people():
    return jsonify({"people": list_people()})

@app.route("/api/reload_faces", methods=["POST"])
def api_reload_faces():
    engine.reload_faces()
    return jsonify({"ok": True, "identities": len(engine.known_embeddings)})

@app.route("/api/upload_face", methods=["POST"])
def api_upload_face():
    """
    form-data:
      - mode: "existing" | "new"
      - existing_name: existing folder name (optional)
      - new_name: new folder name (optional)
      - file: image file
    """
    f = request.files.get("file")
    mode = (request.form.get("mode") or "").strip().lower()
    existing_name = (request.form.get("existing_name") or "").strip()
    new_name = (request.form.get("new_name") or "").strip()

    if not f:
        return jsonify({"ok": False, "error": "file is required"}), 400

    # pick person name
    if mode == "existing":
        person = safe_person_name(existing_name)
    else:
        person = safe_person_name(new_name)

    if not person:
        return jsonify({"ok": False, "error": "person name is required"}), 400

    # extension
    original = secure_filename(f.filename or "")
    ext = os.path.splitext(original)[1].lower()
    if ext not in ALLOWED_EXTS:
        # default if missing or unsupported
        ext = ".jpg"

    person_dir = os.path.join(FACES_DIR, person)
    os.makedirs(person_dir, exist_ok=True)

    filename = next_image_filename(person_dir, ext)
    path = os.path.join(person_dir, filename)
    f.save(path)

    # Add some delay before reloading faces to allow file save to complete
    time.sleep(1)  # You can adjust this based on file sizes

    engine.reload_faces()
    return jsonify({"ok": True, "person": person, "saved": f"/faces/{person}/{filename}"})

@app.route("/faces/<person>/<path:filename>")
def faces_file(person, filename):
    # prevent path traversal / weird names
    person = safe_person_name(person)
    if not person:
        return ("", 404)

    # serve actual file
    return send_from_directory(os.path.join(FACES_DIR, person), filename)

@app.route("/screenshots/<path:filename>")
def screenshot_file(filename):
    """Serve visit screenshot images."""
    return send_from_directory(SCREENSHOTS_DIR, filename)

@app.route("/footage/<path:filename>")
def footage_file(filename):
    """Serve visit footage video clips (supports Range requests for streaming)."""
    return send_from_directory(FOOTAGE_DIR, filename, conditional=True)

@app.route("/api/status")
def api_status():
    return jsonify({
        "running": engine.is_running(),
        "identities": len(engine.known_embeddings),
        "camera_source": _camera_source_to_text(engine.cam_index),
    })


@app.route("/api/camera", methods=["GET"])
def api_camera_get():
    return jsonify(
        {
            "running": engine.is_running(),
            "camera_source": _camera_source_to_text(engine.cam_index),
            "devices": _list_camera_devices(),
        }
    )


@app.route("/api/camera", methods=["POST"])
def api_camera_set():
    payload = request.get_json(silent=True) or {}
    source_raw = payload.get("source", request.form.get("source", ""))
    source_text = str(source_raw or "").strip()
    if not source_text:
        return jsonify({"ok": False, "error": "source is required"}), 400

    restart = bool(payload.get("restart", True))

    # Match grid_RxC patterns (e.g. grid_3x2, grid_4x4)
    grid_match = re.fullmatch(r"grid_(\d+)x(\d+)", source_text)
    if grid_match:
        rows, cols = int(grid_match.group(1)), int(grid_match.group(2))
        try:
            engine.set_grid_layout(rows, cols)
        except ValueError as e:
            return jsonify({"ok": False, "error": str(e)}), 400

        # Check if we have a saved grid config for this layout
        saved = FaceEngine.load_grid_config()
        if saved and tuple(saved["layout"]) == (rows, cols):
            new_source = saved["cam_index"]
        else:
            # Auto-fill: discover devices and fill slots sequentially
            numeric_sources = []
            for d in _list_camera_devices():
                val = str(d.get("value", "")).strip()
                if re.fullmatch(r"\d+", val):
                    numeric_sources.append(val)
            if not numeric_sources:
                return jsonify({"ok": False, "error": "no camera devices found for grid mode"}), 400
            max_slots = rows * cols
            new_source = "grid:" + ",".join(numeric_sources[:max_slots])
    else:
        new_source = int(source_text) if re.fullmatch(r"\d+", source_text) else source_text

    with state_lock:
        old_source = engine.cam_index
        was_running = engine.is_running()
        source_changed = new_source != old_source

        if source_changed and was_running:
            engine.stop()
        if source_changed:
            engine.cam_index = new_source

        should_start = (was_running or restart) and (not engine.is_running())
        if should_start:
            try:
                engine.start()
            except Exception as e:
                if engine.is_running():
                    engine.stop()
                engine.cam_index = old_source
                restored = False
                if was_running:
                    try:
                        engine.start()
                        restored = True
                    except Exception:
                        restored = False
                return jsonify(
                    {
                        "ok": False,
                        "error": f"failed to open camera source: {e}",
                        "camera_source": _camera_source_to_text(engine.cam_index),
                        "running": engine.is_running(),
                        "restored_previous_source": restored,
                    }
                ), 500

    if source_changed:
        with attendance_lock:
            _mark_all_absent()

    return jsonify(
        {
            "ok": True,
            "running": engine.is_running(),
            "camera_source": _camera_source_to_text(engine.cam_index),
            "devices": _list_camera_devices(),
        }
    )

@app.route("/api/grid/config", methods=["GET"])
def api_grid_config_get():
    """Return current grid layout, slot assignments, and available options."""
    saved = FaceEngine.load_grid_config()
    rows, cols = engine._grid_layout
    max_slots = rows * cols

    # Build current slots from saved config or from the live cam_index
    if saved and tuple(saved["layout"]) == (rows, cols):
        slots = saved["slots"]
    else:
        # Derive from current cam_index if in grid mode
        slots = {}
        if engine._is_grid_mode():
            sources = FaceEngine._parse_grid_sources(engine.cam_index)
            for i, src in enumerate(sources[:max_slots]):
                slots[str(i)] = FaceEngine._normalize_slot(src)
        # Pad remaining slots
        for i in range(max_slots):
            if str(i) not in slots:
                slots[str(i)] = None

    # Available cameras (exclude grid options)
    cameras = []
    for d in _list_camera_devices():
        val = str(d.get("value", "")).strip()
        if val.startswith("grid_"):
            continue
        cameras.append({"value": val, "label": d.get("label", val)})

    return jsonify({
        "layout": [rows, cols],
        "slots": slots,
        "available_layouts": [[r, c] for r, c in AVAILABLE_LAYOUTS],
        "available_cameras": cameras,
    })


@app.route("/api/grid/config", methods=["POST"])
def api_grid_config_set():
    """Set grid layout and slot assignments, save to disk, and restart."""
    payload = request.get_json(silent=True) or {}
    layout = payload.get("layout")
    slots = payload.get("slots")

    if not layout or not isinstance(layout, (list, tuple)) or len(layout) != 2:
        return jsonify({"ok": False, "error": "layout must be [rows, cols]"}), 400
    if not isinstance(slots, dict):
        return jsonify({"ok": False, "error": "slots must be an object"}), 400

    rows, cols = int(layout[0]), int(layout[1])
    try:
        engine.set_grid_layout(rows, cols)
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    max_slots = rows * cols
    # Validate and clean slots (new format: {source, name} objects)
    clean_slots = {}
    for i in range(max_slots):
        val = slots.get(str(i))
        normalized = FaceEngine._normalize_slot(val)
        clean_slots[str(i)] = normalized

    # Reject duplicate camera assignments (compare source values)
    assigned = [v["source"] for v in clean_slots.values() if v is not None]
    dupes = [v for v in assigned if assigned.count(v) > 1]
    if dupes:
        seen = set(dupes)
        return jsonify({
            "ok": False,
            "error": f"Each camera can only be assigned to one slot. Duplicates: {', '.join(sorted(seen))}",
        }), 400

    # Save to disk
    try:
        FaceEngine.save_grid_config((rows, cols), clean_slots)
    except Exception as e:
        return jsonify({"ok": False, "error": f"failed to save config: {e}"}), 500

    # Sync locations to DB
    if db.is_available():
        for src, name in FaceEngine.get_slot_locations(clean_slots).items():
            db.upsert_location(src, name)

    # Build cam_index and restart
    cam_index = FaceEngine.build_grid_cam_index(clean_slots)

    with state_lock:
        was_running = engine.is_running()
        if was_running:
            engine.stop()
        engine.cam_index = cam_index
        if was_running:
            try:
                engine.start()
            except Exception as e:
                return jsonify({
                    "ok": False,
                    "error": f"failed to start with new grid config: {e}",
                    "running": engine.is_running(),
                }), 500

    with attendance_lock:
        _mark_all_absent()

    return jsonify({
        "ok": True,
        "running": engine.is_running(),
        "layout": [rows, cols],
        "slots": clean_slots,
        "camera_source": _camera_source_to_text(engine.cam_index),
    })


# ---------------------------------------------------------------------------
# History / report endpoints
# ---------------------------------------------------------------------------

@app.route("/history")
def history_page():
    return render_template("history.html")


def _to_dt(val):
    """Convert a value to datetime — handles both datetime objects and ISO strings."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    if isinstance(val, str):
        return datetime.fromisoformat(val)
    return val


def _serialize_visit(v):
    """Convert a visit row (dict) to a JSON-safe dict with duration."""
    first = _to_dt(v["first_seen"])
    last = _to_dt(v["last_seen"])
    duration_secs = (last - first).total_seconds() if first and last else 0
    return {
        "id": v.get("id"),
        "person_name": v.get("person_name", ""),
        "location_name": v.get("location_name", ""),
        "camera_source": v.get("camera_source", ""),
        "first_seen": first.isoformat() if first else None,
        "last_seen": last.isoformat() if last else None,
        "duration_secs": round(duration_secs, 1),
        "duration_fmt": _fmt_duration(duration_secs),
        "ended": bool(v.get("ended", False)),
        "confidence": round(float(v["confidence"]), 3) if v.get("confidence") else None,
        "screenshot_url": f"/screenshots/{v['screenshot']}" if v.get("screenshot") else None,
        "footage_url": f"/footage/{v['footage']}" if v.get("footage") else None,
    }


def _fmt_duration(secs):
    """Format seconds into 'Xh Ym' or 'Xm Ys'."""
    secs = max(0, int(secs))
    if secs < 60:
        return f"{secs}s"
    mins = secs // 60
    rem_secs = secs % 60
    if mins < 60:
        return f"{mins}m {rem_secs}s"
    hours = mins // 60
    rem_mins = mins % 60
    return f"{hours}h {rem_mins}m"


@app.route("/api/history/daily")
def api_history_daily():
    """Daily summary. Query param: ?date=YYYY-MM-DD (defaults to today)."""
    if not db.is_available():
        return jsonify({"ok": False, "error": "database not available"}), 503
    date_str = request.args.get("date")
    try:
        day = datetime.strptime(date_str, "%Y-%m-%d").date() if date_str else date.today()
    except ValueError:
        return jsonify({"ok": False, "error": "invalid date format, use YYYY-MM-DD"}), 400

    visits = db.get_daily_summary(day)
    rows = [_serialize_visit(v) for v in visits]

    # Group by person for the summary
    person_totals = {}
    for r in rows:
        name = r["person_name"]
        if name not in person_totals:
            person_totals[name] = 0.0
        person_totals[name] += r["duration_secs"]

    return jsonify({
        "ok": True,
        "date": day.isoformat(),
        "visits": rows,
        "person_totals": {k: {"total_secs": v, "total_fmt": _fmt_duration(v)} for k, v in person_totals.items()},
    })


@app.route("/api/history/person/<name>")
def api_history_person(name):
    """All visits for a person. Query params: ?from=YYYY-MM-DD&to=YYYY-MM-DD"""
    if not db.is_available():
        return jsonify({"ok": False, "error": "database not available"}), 503
    date_from = request.args.get("from")
    date_to = request.args.get("to")
    try:
        df = datetime.strptime(date_from, "%Y-%m-%d").replace(tzinfo=timezone.utc) if date_from else None
        dt = (datetime.strptime(date_to, "%Y-%m-%d") + timedelta(days=1)).replace(tzinfo=timezone.utc) if date_to else None
    except ValueError:
        return jsonify({"ok": False, "error": "invalid date format"}), 400

    visits = db.get_person_visits(name, date_from=df, date_to=dt)
    rows = [_serialize_visit(v) for v in visits]
    total_secs = sum(r["duration_secs"] for r in rows)
    return jsonify({
        "ok": True,
        "person_name": name,
        "visits": rows,
        "total_secs": round(total_secs, 1),
        "total_fmt": _fmt_duration(total_secs),
    })


@app.route("/api/history/location/<int:location_id>")
def api_history_location(location_id):
    """All visits at a location. Query params: ?from=YYYY-MM-DD&to=YYYY-MM-DD"""
    if not db.is_available():
        return jsonify({"ok": False, "error": "database not available"}), 503
    date_from = request.args.get("from")
    date_to = request.args.get("to")
    try:
        df = datetime.strptime(date_from, "%Y-%m-%d").replace(tzinfo=timezone.utc) if date_from else None
        dt = (datetime.strptime(date_to, "%Y-%m-%d") + timedelta(days=1)).replace(tzinfo=timezone.utc) if date_to else None
    except ValueError:
        return jsonify({"ok": False, "error": "invalid date format"}), 400

    visits = db.get_location_visits(location_id, date_from=df, date_to=dt)
    rows = [_serialize_visit(v) for v in visits]
    return jsonify({"ok": True, "location_id": location_id, "visits": rows})


@app.route("/api/history/locations")
def api_history_locations():
    """List all known locations."""
    if not db.is_available():
        return jsonify({"ok": False, "error": "database not available"}), 503
    locs = db.get_locations()
    return jsonify({"ok": True, "locations": locs})


@app.route("/api/history/persons")
def api_history_persons():
    """List all known person names with visits."""
    if not db.is_available():
        return jsonify({"ok": False, "error": "database not available"}), 503
    return jsonify({"ok": True, "persons": db.get_known_persons()})


@app.route("/api/history/sessions")
def api_history_sessions():
    if not db.is_available():
        return jsonify({"ok": False, "error": "database not available"}), 503
    sessions = db.get_sessions()
    rows = []
    for s in sessions:
        started = s["started_at"]
        ended = s["ended_at"]
        # Handle both datetime objects (Postgres) and ISO strings (SQLite)
        if isinstance(started, str):
            started_dt = datetime.fromisoformat(started) if started else None
        else:
            started_dt = started
        if isinstance(ended, str):
            ended_dt = datetime.fromisoformat(ended) if ended else None
        else:
            ended_dt = ended
        duration = (ended_dt - started_dt).total_seconds() if started_dt and ended_dt else None
        rows.append({
            "id": str(s["id"]),
            "started_at": started_dt.isoformat() if started_dt else None,
            "ended_at": ended_dt.isoformat() if ended_dt else None,
            "duration_fmt": _fmt_duration(duration) if duration else "running",
            "camera_source": s.get("camera_source") if isinstance(s, dict) else None,
        })
    return jsonify({"ok": True, "sessions": rows})


@app.route("/api/history/clear", methods=["POST"])
def api_history_clear():
    """Delete all visits and sessions, reset in-memory tracking state."""
    if not db.is_available():
        return jsonify({"ok": False, "error": "database not available"}), 503
    global _current_session_id
    engine.stop_all_footage()
    visit_count = db.clear_all_data()
    _active_visits.clear()
    attendance_state.clear()
    # Clear screenshot files
    for f in os.listdir(SCREENSHOTS_DIR):
        try:
            os.remove(os.path.join(SCREENSHOTS_DIR, f))
        except Exception:
            pass
    # Clear footage files
    for f in os.listdir(FOOTAGE_DIR):
        try:
            os.remove(os.path.join(FOOTAGE_DIR, f))
        except Exception:
            pass
    # Start a fresh session
    _current_session_id = db.create_session(
        camera_source=str(engine.cam_index) if engine.cam_index is not None else None
    )
    return jsonify({"ok": True, "deleted_visits": visit_count})


@app.route("/api/start", methods=["POST"])
def api_start():
    with state_lock:
        if not engine.is_running():
            engine.start()
    return jsonify({"ok": True, "running": engine.is_running()})

@app.route("/api/stop", methods=["POST"])
def api_stop():
    with state_lock:
        if engine.is_running():
            engine.stop()

    with attendance_lock:
        _mark_all_absent()

    return jsonify({"ok": True, "running": engine.is_running()})

@app.route("/api/attendance/stream")
def attendance_stream():
    def gen():
        # Send initial snapshot immediately
        with attendance_lock:
            running = engine.is_running()
            tracks = engine.get_tracks() if running else []
            _update_qr_prompt_from_tracks(tracks, running)
            prompt_payload = _get_qr_prompt_payload()
            payload = {
                "running": running,
                "attendance": _attendance_roster(),
                "qr_prompt": prompt_payload["qr_prompt"],
                "unknown_elapsed_secs": prompt_payload["unknown_elapsed_secs"],
            }
        yield f"event: state\ndata: {json.dumps(payload)}\n\n"

        while True:
            try:
                msg = attendance_events_q.get(timeout=15)
            except Empty:
                yield ": ping\n\n"  # keepalive
                continue

            yield f"event: {msg['event']}\ndata: {json.dumps(msg['data'])}\n\n"

    return Response(
        stream_with_context(gen()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    # IMPORTANT: don’t use the reloader with a webcam engine (it runs twice)
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True, use_reloader=False)
