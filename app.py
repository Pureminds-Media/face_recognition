import os
import time
import re
import json
import glob
import subprocess
from queue import Queue, Empty, Full
from flask import Flask, Response, render_template, request, jsonify, send_from_directory, stream_with_context
from werkzeug.utils import secure_filename
import threading
from face_engine import FaceEngine

app = Flask(__name__)

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
FACES_DIR = "faces"

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

# Start engine once (avoid double-start in debug reloader)
engine.start()
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
qr_prompt_state = {
    "unknown_first_mono": None,
    "unknown_last_seen_mono": None,
    "active": False,
}

def _q_put_latest(q, item):
    try:
        q.put_nowait(item)
    except Full:
        try:
            q.get_nowait()
        except Empty:
            pass
        q.put_nowait(item)

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
    """Update attendance_state based on current recognized tracks.

    - First time a person is seen: mark attended once and emit a 'new' event
    - If person disappears (not seen for ATTENDANCE_DISAPPEAR_SECS) and later reappears: emit a 'repeat' event
    """
    now_mono = time.monotonic()
    now_ts = time.time()
    events = []

    for t in (tracks or []):
        name = (t or {}).get("name")
        if not name or name == "unknown":
            continue

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

    # Mark disappeared after a grace period (avoid flicker)
    for name, s in list(attendance_state.items()):
        if s.get("present", False) and (now_mono - s.get("last_seen_mono", now_mono)) > ATTENDANCE_DISAPPEAR_SECS:
            s["present"] = False

    return events

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
    return str(source) if source is not None else ""


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

@app.route("/video")
def video():
    if not engine.is_running():
        return ("Camera stopped", 503)

    return Response(
        mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

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
