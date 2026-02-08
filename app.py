import os
import time
import re
import json
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
    detect_every=1.0,
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
    return roster

def _attendance_loop():
    last_sig = None
    last_running = None

    while True:
        running = engine.is_running()
        events = []

        with attendance_lock:
            if running:
                events = _update_attendance_from_tracks(engine.get_tracks())
            else:
                if last_running:
                    _mark_all_absent()

            roster = _attendance_roster()
            sig = (
                running,
                tuple((r["name"], r["attended"], r["present"]) for r in roster),
            )

        # Emit state only if it changed OR if we have events
        if sig != last_sig or events:
            _q_put_latest(attendance_events_q, {"event": "state", "data": {"running": running, "attendance": roster}})
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

threading.Thread(target=_attendance_loop, daemon=True).start()

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


def mjpeg_generator():
    # If camera stops mid-stream, exit generator
    while engine.is_running():
        frame = engine.get_jpeg()
        if frame is None:
            time.sleep(0.05)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame
            + b"\r\n"
        )

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
        events = _update_attendance_from_tracks(engine.get_tracks()) if running else []
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
    })

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
            payload = {"running": engine.is_running(), "attendance": _attendance_roster()}
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