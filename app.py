import os
# RTSP/HTTP camera tuning for OpenCV's FFmpeg backend.
# - rtsp_transport=tcp avoids UDP packet loss/NAT issues common on Wi-Fi cams.
# - stimeout (microseconds) caps socket-level read waits so a dead camera
#   surfaces as a clean failure instead of hanging.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|stimeout;5000000",
)
# Silence the HEVC mid-GOP join noise ("PPS id out of range",
# "Could not find ref with POC N", "First slice in a frame missing",
# "Error constructing the frame RPS"). These are AV_LOG_ERROR level
# even though they're benign chatter while the decoder waits for the
# next keyframe; with 21 RTSP streams they flood the terminal. We drop
# to AV_LOG_FATAL (8) — actual fatal decoder errors still surface,
# everything else is muted.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "8")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
import shutil
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
# Backstop the env-var settings above in case this OpenCV build ignores
# them. LOG_LEVEL_ERROR=4 is the standard symbol; fall back to the int
# constant if the symbolic name isn't present.
try:
    cv2.setLogLevel(getattr(cv2, "LOG_LEVEL_ERROR", 4))
except Exception:
    pass
from dotenv import load_dotenv
from face_engine import FaceEngine, AVAILABLE_LAYOUTS  # FaceEngine kept for static helpers
from engine_client import EngineClient
import multiprocessing as _mp
import engine_runner as _engine_runner
import db

load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="[web]    %(asctime)s %(levelname)s %(message)s",
)
# Suppress "Client disconnected while serving …" noise — these fire every
# time a browser closes an SSE/MJPEG/footage stream, which is normal.
logging.getLogger("werkzeug").addFilter(
    lambda r: "Client disconnected" not in r.getMessage()
)
logging.getLogger("waitress").addFilter(
    lambda r: "Client disconnected" not in r.getMessage()
)
log = logging.getLogger(__name__)

app = Flask(__name__)


@app.errorhandler(RuntimeError)
def _handle_engine_offline(e):
    if "Engine process is not running" in str(e):
        return jsonify({"ok": False, "error": "Engine process is not running"}), 503
    raise e


# --- Public API auth ---------------------------------------------------------
# When API_KEY is set, every /api/* and stream route requires the same key
# in either an X-API-Key header or an api_key query string. Page routes
# (HTML templates) stay open so the locally served UI keeps working without
# a cookie/session layer. The local UI auto-attaches the key (injected into
# the templates via render_template); external clients must send it themselves.
API_KEY = os.getenv("API_KEY", "").strip()

_PROTECTED_PREFIXES = (
    "/api/", 
    "/video", 
    "/footage/", 
    # "/faces/"
    )


_LOCAL_HOSTS = {"localhost", "127.0.0.1", "[::1]", "::1"}


def _is_local_request():
    """True when the client connected to the local Flask socket directly
    (e.g. curl on the host, the same-machine browser). ngrok rewrites the
    Host header to its public hostname so tunneled traffic does NOT match
    here, which is what we want — the API key is still required externally.
    """
    host = (request.headers.get("Host") or "").split(":", 1)[0].lower()
    return host in _LOCAL_HOSTS


@app.before_request
def _require_api_key():
    """Gate protected routes on X-API-Key when API_KEY is configured.

    Skipped for requests with a localhost Host header so curl on the host
    and the locally served browser UI work without supplying the key.
    """
    if not API_KEY:
        return None
    if _is_local_request():
        return None
    path = request.path or ""
    if not path.startswith(_PROTECTED_PREFIXES):
        return None
    supplied = request.headers.get("X-API-Key") or request.args.get("api_key") or ""
    if supplied != API_KEY:
        return jsonify({"ok": False, "error": "unauthorized"}), 401
    return None


@app.context_processor
def _inject_api_key():
    """Make API_KEY available to Jinja templates so the local UI can attach
    it to every fetch() automatically."""
    return {"API_KEY": API_KEY}

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
FACES_DIR = "faces"
TEST_UPLOAD_DIR = os.path.join("test_runs", "uploads")
TEST_OUTPUT_DIR = os.path.join("test_runs", "outputs")
FOOTAGE_DIR = os.environ["FOOTAGE_DIR"]
IP_CAMERAS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ip_cameras.json")
_ip_cameras_lock = threading.Lock()


def _new_id():
    return uuid.uuid4().hex[:8]


def _normalize_ip_cameras(data):
    """Return ``(state, migrated)``. ``migrated`` is True when *data* was
    in an older format and we converted it; the caller should persist the
    state so subsequent reads see stable IDs."""
    # Legacy: bare list of cameras with full URLs.
    if isinstance(data, list):
        cams = [c for c in data if isinstance(c, dict) and c.get("id") and c.get("url")]
        return ({
            "groups": [{
                "id": _new_id(),
                "name": "Standalone",
                "base_url": "",
                "cameras": cams,
            }] if cams else []
        }, True)

    if not isinstance(data, dict):
        return ({"groups": []}, True)

    # Earlier single-base format.
    if "groups" not in data and ("base_url" in data or "cameras" in data):
        cams = [c for c in (data.get("cameras") or []) if isinstance(c, dict) and c.get("id")]
        return ({
            "groups": [{
                "id": _new_id(),
                "name": "Default",
                "base_url": str(data.get("base_url") or "").strip(),
                "cameras": cams,
            }] if cams else []
        }, True)

    # Current multi-group format.
    groups = []
    for g in data.get("groups") or []:
        if not isinstance(g, dict) or not g.get("id"):
            continue
        cams = []
        for c in g.get("cameras") or []:
            if not isinstance(c, dict) or not c.get("id"):
                continue
            if c.get("channel") or c.get("url"):
                cams.append(c)
        groups.append({
            "id": str(g["id"]),
            "name": str(g.get("name") or "Group").strip(),
            "base_url": str(g.get("base_url") or "").strip(),
            "cameras": cams,
        })
    return ({"groups": groups}, False)


def _load_ip_cameras():
    """Return the IP-camera config dict. Migrates older formats on the
    fly and persists the migration so IDs are stable across requests.
    """
    try:
        with open(IP_CAMERAS_PATH) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"groups": []}

    state, migrated = _normalize_ip_cameras(data)
    if migrated:
        # Persist immediately. Without this, every subsequent read would
        # mint fresh UUIDs and the frontend's PUT/DELETE calls (which
        # reference the IDs the browser saw on its last render) would
        # 404 with "group not found".
        try:
            _save_ip_cameras(state)
        except Exception:
            pass
    return state


def _save_ip_cameras(state):
    with open(IP_CAMERAS_PATH, "w") as f:
        json.dump(state, f, indent=2)


def _resolved_camera_url(cam, base_url):
    """Compute the full RTSP URL. Returns "" if incomplete."""
    if cam.get("url"):
        return str(cam["url"]).strip()
    ch = str(cam.get("channel") or "").strip()
    if ch and base_url:
        return base_url + ch
    return ""


def _find_group(state, group_id):
    return next((g for g in state.get("groups", []) if g.get("id") == group_id), None)


def _find_camera(state, camera_id):
    """Return (group, camera) or (None, None)."""
    for g in state.get("groups", []):
        for c in g.get("cameras", []):
            if c.get("id") == camera_id:
                return g, c
    return None, None


def _expanded_cameras(state):
    """Yield (group, cam, resolved_url) for every camera that has a URL."""
    for g in state.get("groups", []):
        base = g.get("base_url", "")
        for c in g.get("cameras", []):
            url = _resolved_camera_url(c, base)
            if not url:
                continue
            yield g, c, url


def _serialize_state(state):
    """Frontend-shaped view: each camera carries `resolved_url`."""
    out = {"groups": []}
    for g in state.get("groups", []):
        cams_out = []
        base = g.get("base_url", "")
        for c in g.get("cameras", []):
            cams_out.append({
                "id": c.get("id"),
                "name": c.get("name") or "",
                "channel": c.get("channel") or "",
                "url": c.get("url") or "",
                "resolved_url": _resolved_camera_url(c, base),
            })
        out["groups"].append({
            "id": g.get("id"),
            "name": g.get("name") or "",
            "base_url": base,
            "cameras": cams_out,
        })
    return out
ALLOWED_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

os.makedirs(FOOTAGE_DIR, exist_ok=True)

# The engine runs in a subprocess. We can't spawn it at module-import time
# because Python's `spawn` start method re-imports app.py in the child;
# any unguarded mp.Process() / Manager() at module level recurses.
# Instead, expose a bootstrap function that the __main__ block calls.
def _env_bool(name, default):
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off", "")


_ENGINE_KWARGS = dict(
    known_dir=FACES_DIR,
    detector="retinaface",
    model="buffalo_l",
    threshold=0.55,
    detect_every=5,
    detect_scale=1,
    tracker_type="CSRT",
    width=1280,
    height=720,
    out_fps=10,
    jpeg_quality=80,
    # Draw bounding boxes + name labels on the live MJPEG feed.
    # Default on. Set LIVE_ANNOTATIONS_ENABLED=0 in .env to get a clean
    # stream — useful when many people are in frame and box overlays
    # become illegible. Footage recordings and per-visit screenshots
    # are always annotated regardless of this flag.
    live_annotations=_env_bool("LIVE_ANNOTATIONS_ENABLED", True),
)

# Module-level placeholder so route handlers (which only resolve `engine`
# at request time) can be defined before bootstrap runs.
engine = None
_engine_manager = None
_engine_state = None
_engine_proc = None
_engine_parent_conn = None


def _spawn_engine_proc():
    """Create and start a fresh engine subprocess. Returns (proc, parent_conn)."""
    parent_conn, child_conn = _mp.Pipe()
    proc = _mp.Process(
        target=_engine_runner.run,
        args=(child_conn, _engine_state, _ENGINE_KWARGS),
        daemon=False,
        name="face-engine",
    )
    proc.start()
    return proc, parent_conn


def _bootstrap_engine():
    """Spawn the engine subprocess and bind the EngineClient.

    Called once from the __main__ block. Idempotent: subsequent calls
    are no-ops so any accidental double-call (e.g. waitress reload) is
    safe.
    """
    global engine, _engine_manager, _engine_state, _engine_proc, _engine_parent_conn
    if engine is not None:
        return

    try:
        _mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # already set

    _engine_manager = _mp.Manager()
    _engine_state = _engine_manager.dict()

    _engine_proc, parent_conn = _spawn_engine_proc()
    _engine_parent_conn = parent_conn
    engine = EngineClient(parent_conn, _engine_state, threading.Lock())

    import atexit as _atexit
    def _shutdown_engine():
        try: _engine_parent_conn.send("shutdown")
        except Exception: pass
        try: _engine_proc.join(timeout=5)
        except Exception: pass
        if _engine_proc.is_alive():
            _engine_proc.terminate()
    _atexit.register(_shutdown_engine)
    # Note: _shutdown_engine reads the globals _engine_parent_conn and
    # _engine_proc at call time, so it always targets the current process
    # even if the watchdog has restarted it.

    log.info("waiting for engine subprocess to publish state...")
    t0 = time.time()
    while "cam_index" not in _engine_state and (time.time() - t0) < 30:
        time.sleep(0.1)
    if "cam_index" not in _engine_state:
        log.warning("engine subprocess slow to boot — continuing with defaults "
                    "(state keys so far: %s)", list(_engine_state.keys()))
    else:
        log.info("engine ready, cam_index=%s, identities=%s",
                 _engine_state.get("cam_index"), _engine_state.get("identities"))

    threading.Thread(target=_engine_watchdog, daemon=True, name="engine-watchdog").start()


def _engine_watchdog():
    """Restart the engine subprocess if it dies unexpectedly."""
    global _engine_proc, _engine_parent_conn
    while True:
        time.sleep(5)
        try:
            if _engine_proc is not None and not _engine_proc.is_alive():
                exit_code = _engine_proc.exitcode
                log.warning("engine subprocess died (exit code %s) — restarting", exit_code)
                try: _engine_proc.join(timeout=2)
                except Exception: pass

                new_proc, new_conn = _spawn_engine_proc()
                _engine_proc = new_proc
                _engine_parent_conn = new_conn

                # Swap the connection on the existing EngineClient so all
                # existing route handlers pick it up without a restart.
                object.__setattr__(engine, "_conn", new_conn)
                object.__setattr__(engine, "_lock", threading.Lock())

                # Wait for the new process to publish its state
                t0 = time.time()
                while "cam_index" not in _engine_state and (time.time() - t0) < 30:
                    time.sleep(0.1)
                log.info("engine subprocess restarted (cam_index=%s)",
                         _engine_state.get("cam_index"))

                # Auto-resume the camera grid so the user doesn't have to
                # manually restart after a crash.
                try:
                    with state_lock:
                        if not engine.is_running():
                            pool = _build_analysis_pool_source()
                            if pool:
                                engine.cam_index = pool
                            _refresh_source_name_map()
                            engine.start()
                            log.info("engine watchdog: auto-resumed camera grid")
                except Exception:
                    log.exception("engine watchdog: auto-resume failed")
        except Exception:
            log.exception("engine watchdog error")

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
VISIT_TRANSITION_SECS = float(os.getenv("VISIT_TRANSITION_SECS", "30.0"))  # seconds absent from current camera before transitioning to a new location
qr_prompt_state = {
    "unknown_first_mono": None,
    "unknown_last_seen_mono": None,
    "active": False,
}
test_jobs_lock = threading.Lock()
test_job_run_lock = threading.Lock()
test_jobs = {}  # job_id -> {status, progress, error, result_url, ...}

# --- Database init ---
# init_db is engine-independent; run at import. Session bootstrap (which
# needs engine.cam_index) runs from _bootstrap_post_engine() below.
db.init_db()
_current_session_id = None


def _bootstrap_post_engine():
    """Engine-dependent boot steps. Called from __main__ after the engine
    subprocess is up. Sets the DB session and starts the attendance / QR
    background loops that read tracks from the engine."""
    global _current_session_id
    if db.is_available() and _current_session_id is None:
        _current_session_id = db.create_session(
            camera_source=str(engine.cam_index) if engine.cam_index is not None else None
        )
        _saved_grid = FaceEngine.load_grid_config()
        if _saved_grid and _saved_grid.get("slots"):
            for _src, _name in FaceEngine.get_slot_locations(_saved_grid["slots"]).items():
                db.upsert_location(_src, _name)
        if not engine._is_grid_mode():
            _src = str(engine.cam_index)
            db.upsert_location(_src, f"Camera {_src}")
        log.info("DB session started: %s", _current_session_id)
    threading.Thread(target=_attendance_loop, daemon=True).start()
    threading.Thread(target=_qr_loop, daemon=True).start()

# In-memory visit tracking state: person_name -> {location_id, visit_id, camera_source}
_active_visits = {}
_last_stale_check = time.monotonic()

# --- Background DB writer ---------------------------------------------------
# Per-frame DB writes (update_visit_seen, update_visit_activity) used to run
# inline on the attendance update path, blocking frame encoding when the WAL
# flushed or the disk was busy. Instead we throttle them to once every
# DB_SEEN_THROTTLE_SECS per visit and drain them off-thread.
DB_SEEN_THROTTLE_SECS = 2.0
_db_writer_q: "Queue[tuple]" = Queue(maxsize=10000)
_last_seen_write: dict = {}      # visit_id -> last update_visit_seen monotonic
_last_activity_write: dict = {}  # visit_id -> last update_visit_activity monotonic


def _db_writer_loop():
    while True:
        try:
            op = _db_writer_q.get()
        except Exception:
            continue
        if op is None:
            return
        try:
            kind = op[0]
            if kind == "seen":
                _, vid, conf = op
                db.update_visit_seen(vid, confidence=conf)
            elif kind == "activity":
                _, vid, label = op
                db.update_visit_activity(vid, label)
        except Exception as e:
            log.debug("DB writer op failed: %s", e)


threading.Thread(target=_db_writer_loop, daemon=True, name="db-writer").start()


def _enqueue_visit_seen(visit_id, confidence=None):
    """Throttled, non-blocking update_visit_seen."""
    if visit_id is None:
        return
    now = time.monotonic()
    last = _last_seen_write.get(visit_id, 0.0)
    if (now - last) < DB_SEEN_THROTTLE_SECS:
        return
    _last_seen_write[visit_id] = now
    try:
        _db_writer_q.put_nowait(("seen", visit_id, confidence))
    except Full:
        pass


def _enqueue_visit_activity(visit_id, label):
    """Throttled, non-blocking update_visit_activity."""
    if visit_id is None or not label:
        return
    now = time.monotonic()
    last = _last_activity_write.get(visit_id, 0.0)
    if (now - last) < DB_SEEN_THROTTLE_SECS:
        return
    _last_activity_write[visit_id] = now
    try:
        _db_writer_q.put_nowait(("activity", visit_id, label))
    except Full:
        pass


def _shutdown_db():
    # Save activity for all active visits before closing
    if db.is_available():
        for name, v in list(_active_visits.items()):
            vid = v.get("visit_id")
            if vid:
                top = engine.get_visit_top_activity(vid)
                if top:
                    try:
                        db.update_visit_activity(vid, top)
                    except Exception:
                        pass
                engine.clear_visit_activity(vid)
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
        last_head_t = float((t or {}).get("last_head_t", 0.0))

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
            _update_visit_for_person(name, camera_source, confidence, last_detect_t, last_head_t)

        # --- Record activity tally for this person's active visit ---
        active_v = _active_visits.get(name)
        if active_v and active_v.get("visit_id"):
            act_label, _ = engine.get_activity(name)
            if act_label:
                vid = active_v["visit_id"]
                engine.record_activity_for_visit(vid, act_label)
                # Keep DB column up-to-date (not just on visit close)
                top = engine.get_visit_top_activity(vid)
                if top and db.is_available():
                    _enqueue_visit_activity(vid, top)

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
    """Close the footage VideoWriter for a visit, save visible_duration and activity."""
    try:
        fname, visible_secs = engine.stop_footage(visit_id)
        if fname:
            log.debug("Closed footage writer for visit %s -> %s (%.1fs visible)",
                       visit_id, fname, visible_secs)
            if visible_secs > 0:
                db.update_visit_visible_duration(visit_id, visible_secs)
        # Save the most frequent activity detected during this visit
        top_activity = engine.get_visit_top_activity(visit_id)
        if top_activity:
            db.update_visit_activity(visit_id, top_activity)
        engine.clear_visit_activity(visit_id)
    except Exception as e:
        log.debug("Failed to stop footage for visit %s: %s", visit_id, e)


def _update_visit_for_person(name, camera_source, confidence=None, last_detect_t=0.0, last_head_t=0.0):
    """Open, update, or transition a visit for a person at a camera/location.

    Flip-flop prevention: when a person appears on a *different* camera than
    their current active visit, we do NOT immediately transition.  Instead we
    check how recently they were seen on the *original* camera:

      - If within VISIT_TRANSITION_SECS: they are likely in an overlap
        zone — keep the existing visit, just refresh its DB timestamp.
      - If older than VISIT_TRANSITION_SECS: they have genuinely left the
        original camera — close old visit, open new one at the new location.

    Ghost-box prevention: only detector-confirmed tracks (fresh last_detect_t
    or last_head_t) refresh last_seen_mono.  Tracker-only ghost boxes still
    bump DB last_seen but do NOT prevent visit transitions.
    """
    now_mono = time.monotonic()
    # A track is "confirmed" if the face detector OR head detector saw it
    # recently.  Ghost boxes from CSRT have stale last_detect_t AND last_head_t.
    detect_freshness = getattr(engine, "detect_every", 1.0) * 2.0
    head_freshness = max(3.0, getattr(engine, "detect_every", 1.0) * 5.0)
    is_detector_confirmed = (
        (now_mono - last_detect_t) < detect_freshness
        or (now_mono - last_head_t) < head_freshness
    )

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
        _start_visit_footage(vid, name, camera_source)
    elif active["location_id"] == loc_id:
        # Same location — update last_seen in DB (throttled, off-thread)
        _enqueue_visit_seen(active["visit_id"], confidence=confidence)
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
            _enqueue_visit_seen(active["visit_id"], confidence=confidence)
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

# Threads now started from _bootstrap_post_engine() so they don't run
# in the engine subprocess (which re-imports this module under spawn).

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


_people_cache = {"sig": None, "data": []}
_people_cache_lock = threading.Lock()


def _people_signature():
    """Cheap fingerprint of the faces tree: top-level mtime + per-folder mtimes.

    Changes whenever a person folder is added/removed or any image inside
    one is added/removed/renamed. Avoids the per-image stat the actual
    scan does.
    """
    try:
        os.makedirs(FACES_DIR, exist_ok=True)
        parts = [os.path.getmtime(FACES_DIR)]
        for entry in sorted(os.scandir(FACES_DIR), key=lambda e: e.name):
            if entry.is_dir():
                parts.append(entry.name)
                parts.append(entry.stat().st_mtime)
        return tuple(parts)
    except OSError:
        return None


def list_people():
    sig = _people_signature()
    with _people_cache_lock:
        if sig is not None and sig == _people_cache["sig"]:
            return _people_cache["data"]

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

    with _people_cache_lock:
        _people_cache["sig"] = sig
        _people_cache["data"] = people
    return people


def _resolve_camera_display_name(source):
    """Return a human-friendly camera label (e.g. '2 — Office (IP)').

    Looks up *source* against the configured IP cameras. Never returns the
    raw URL/IP — the visit-history UI must not leak addresses. Falls back
    to 'Unknown camera' when no match is found.
    """
    if not source:
        return "Unknown camera"
    s = str(source)
    try:
        state = _load_ip_cameras()
        for group, cam, url in _expanded_cameras(state):
            if str(url) == s:
                cam_name = str(cam.get("name") or "IP Camera").strip()
                group_name = str(group.get("name") or "").strip()
                if group_name and group_name.lower() != "standalone":
                    return f"{cam_name} — {group_name} (IP)"
                return f"{cam_name} (IP)"
    except Exception:
        pass
    return "Unknown camera"


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

    # Append configured IP cameras (walk every group; standalone or NVR-style).
    state = _load_ip_cameras()
    seen_urls = set()
    for group, cam, url in _expanded_cameras(state):
        if url in seen_urls:
            continue
        seen_urls.add(url)
        cam_name = str(cam.get("name") or "IP Camera").strip()
        group_name = str(group.get("name") or "").strip()
        # Suffix the group name whenever the camera belongs to a meaningful
        # non-standalone group. Older configs store cameras with a raw `url`
        # instead of `channel`, but they're still part of the group.
        in_named_group = bool(group_name) and group_name.lower() != "standalone"
        if in_named_group:
            label = f"{cam_name} — {group_name} (IP)"
        else:
            label = f"{cam_name} (IP)"
        devices.append({"value": url, "label": label, "path": url})

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

@app.route("/settings")
def settings_page():
    return render_template("settings.html")

@app.route("/dashboard")
def dashboard_page():
    return render_template("settings.html")

@app.route("/people")
def people_page():
    return render_template("settings.html")

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

def _reload_faces_async():
    """Fire engine.reload_faces() on a background thread.

    Used by mutating endpoints (upload/move/delete/rename/merge) so the
    HTTP response returns as soon as the filesystem change is committed,
    rather than waiting for embeddings to rebuild. The next detection
    pass picks up the new embeddings once the thread finishes.
    """
    threading.Thread(target=engine.reload_faces, daemon=True).start()


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

    # When creating a new person, check that the folder doesn't already exist
    if mode == "new":
        person_dir_check = os.path.join(FACES_DIR, person)
        if os.path.isdir(person_dir_check):
            return jsonify({"ok": False, "error": f"Person '{person}' already exists. Use 'Existing' to add more images."}), 409

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

    _reload_faces_async()
    return jsonify({"ok": True, "person": person, "saved": f"/faces/{person}/{filename}"})

@app.route("/api/rename_person", methods=["POST"])
def api_rename_person():
    """Rename a person: move faces/ folder + update DB visits."""
    data = request.get_json(silent=True) or {}
    old_name = (data.get("old_name") or "").strip()
    new_name_raw = (data.get("new_name") or "").strip()

    if not old_name:
        return jsonify({"ok": False, "error": "old_name is required"}), 400

    new_name = safe_person_name(new_name_raw)
    if not new_name:
        return jsonify({"ok": False, "error": "new_name is required (letters, numbers, _, -)"}), 400

    old_dir = os.path.join(FACES_DIR, old_name)
    new_dir = os.path.join(FACES_DIR, new_name)

    if not os.path.isdir(old_dir):
        return jsonify({"ok": False, "error": f"Person '{old_name}' not found"}), 404

    if old_name == new_name:
        return jsonify({"ok": True, "person": new_name, "renamed_visits": 0})

    if os.path.exists(new_dir):
        # Merge: move all images from old_dir into new_dir, then delete old_dir.
        import glob as _glob
        moved = 0
        for src_path in _glob.glob(os.path.join(old_dir, "*")):
            fname = os.path.basename(src_path)
            if fname.startswith("."):
                continue  # skip cache files
            dst_path = os.path.join(new_dir, fname)
            # Avoid overwriting — append a suffix if name collides
            if os.path.exists(dst_path):
                base, ext = os.path.splitext(fname)
                dst_path = os.path.join(new_dir, f"{base}_from_{old_name}{ext}")
            shutil.move(src_path, dst_path)
            moved += 1
        shutil.rmtree(old_dir, ignore_errors=True)
        rows = db.rename_person(old_name, new_name)
        _reload_faces_async()
        return jsonify({"ok": True, "person": new_name, "renamed_visits": rows,
                        "merged": True, "images_moved": moved})

    # Simple rename
    os.rename(old_dir, new_dir)
    rows = db.rename_person(old_name, new_name)
    _reload_faces_async()
    return jsonify({"ok": True, "person": new_name, "renamed_visits": rows, "merged": False})

@app.route("/api/person/<name>", methods=["DELETE"])
def api_delete_person(name):
    """Delete a person: remove faces/ folder + optionally delete visits."""
    person = safe_person_name(name)
    if not person:
        return jsonify({"ok": False, "error": "Invalid person name"}), 400

    person_dir = os.path.join(FACES_DIR, person)
    if not os.path.isdir(person_dir):
        return jsonify({"ok": False, "error": f"Person '{person}' not found"}), 404

    # Remove face images
    shutil.rmtree(person_dir)

    # Delete visits from DB
    deleted_visits = db.delete_person_visits(person)

    # Reload face embeddings
    _reload_faces_async()

    return jsonify({"ok": True, "person": person, "deleted_visits": deleted_visits})

@app.route("/api/person/<name>/images")
def api_person_images(name):
    """List all face images for a person."""
    person = safe_person_name(name)
    if not person:
        return jsonify({"ok": False, "error": "Invalid person name"}), 400

    person_dir = os.path.join(FACES_DIR, person)
    if not os.path.isdir(person_dir):
        return jsonify({"ok": False, "error": f"Person '{person}' not found"}), 404

    imgs = []
    for f in sorted(os.listdir(person_dir)):
        if os.path.splitext(f)[1].lower() in ALLOWED_EXTS:
            imgs.append({"filename": f, "url": f"/faces/{person}/{f}"})

    return jsonify({"ok": True, "person": person, "images": imgs})

@app.route("/api/person/<name>/image/<filename>", methods=["DELETE"])
def api_delete_person_image(name, filename):
    """Delete a single face image from a person's folder."""
    person = safe_person_name(name)
    if not person:
        return jsonify({"ok": False, "error": "Invalid person name"}), 400

    # Sanitise filename
    filename = secure_filename(filename)
    if not filename:
        return jsonify({"ok": False, "error": "Invalid filename"}), 400

    person_dir = os.path.join(FACES_DIR, person)
    file_path = os.path.join(person_dir, filename)

    if not os.path.isfile(file_path):
        return jsonify({"ok": False, "error": "Image not found"}), 404

    os.remove(file_path)

    # If directory is now empty, remove the person folder entirely
    remaining = [f for f in os.listdir(person_dir) if os.path.splitext(f)[1].lower() in ALLOWED_EXTS]
    person_removed = False
    if not remaining:
        shutil.rmtree(person_dir)
        person_removed = True

    _reload_faces_async()
    return jsonify({"ok": True, "person_removed": person_removed})

@app.route("/api/person/<name>/image/<filename>/transfer", methods=["POST"])
def api_transfer_person_image(name, filename):
    """Move a face image from one person to another."""
    person = safe_person_name(name)
    if not person:
        return jsonify({"ok": False, "error": "Invalid source person name"}), 400

    filename = secure_filename(filename)
    if not filename:
        return jsonify({"ok": False, "error": "Invalid filename"}), 400

    data = request.get_json(silent=True) or {}
    target_name = safe_person_name(data.get("target", ""))
    if not target_name:
        return jsonify({"ok": False, "error": "target person name is required"}), 400

    if person == target_name:
        return jsonify({"ok": False, "error": "Source and target are the same"}), 400

    src_dir = os.path.join(FACES_DIR, person)
    src_path = os.path.join(src_dir, filename)

    if not os.path.isfile(src_path):
        return jsonify({"ok": False, "error": "Source image not found"}), 404

    # Ensure target directory exists
    tgt_dir = os.path.join(FACES_DIR, target_name)
    os.makedirs(tgt_dir, exist_ok=True)

    # Determine next filename in target folder
    ext = os.path.splitext(filename)[1].lower() or ".jpg"
    new_filename = next_image_filename(tgt_dir, ext)
    tgt_path = os.path.join(tgt_dir, new_filename)

    # Move the file
    shutil.move(src_path, tgt_path)

    # If source directory is now empty of images, remove it
    remaining = [f for f in os.listdir(src_dir) if os.path.splitext(f)[1].lower() in ALLOWED_EXTS] if os.path.isdir(src_dir) else []
    person_removed = False
    if not remaining:
        shutil.rmtree(src_dir, ignore_errors=True)
        person_removed = True

    _reload_faces_async()
    return jsonify({
        "ok": True,
        "target": target_name,
        "new_filename": new_filename,
        "person_removed": person_removed,
    })

@app.route("/api/person/<name>/images/bulk_delete", methods=["POST"])
def api_bulk_delete_person_images(name):
    """Delete multiple images from a person. Body: {"filenames": [...]}."""
    person = safe_person_name(name)
    if not person:
        return jsonify({"ok": False, "error": "Invalid person name"}), 400

    data = request.get_json(silent=True) or {}
    filenames = data.get("filenames") or []
    if not isinstance(filenames, list) or not filenames:
        return jsonify({"ok": False, "error": "filenames list is required"}), 400

    person_dir = os.path.join(FACES_DIR, person)
    if not os.path.isdir(person_dir):
        return jsonify({"ok": False, "error": "Person not found"}), 404

    deleted, missing = [], []
    for raw in filenames:
        fn = secure_filename(str(raw))
        if not fn:
            missing.append(raw)
            continue
        fp = os.path.join(person_dir, fn)
        if not os.path.isfile(fp):
            missing.append(fn)
            continue
        try:
            os.remove(fp)
            deleted.append(fn)
        except Exception:
            missing.append(fn)

    person_removed = False
    remaining = [f for f in os.listdir(person_dir) if os.path.splitext(f)[1].lower() in ALLOWED_EXTS]
    if not remaining:
        shutil.rmtree(person_dir, ignore_errors=True)
        person_removed = True

    _reload_faces_async()
    return jsonify({
        "ok": True, "deleted": deleted, "missing": missing,
        "person_removed": person_removed,
    })


@app.route("/api/person/<name>/images/bulk_transfer", methods=["POST"])
def api_bulk_transfer_person_images(name):
    """Move multiple images to another person. Body: {"target": "...", "filenames": [...]}."""
    person = safe_person_name(name)
    if not person:
        return jsonify({"ok": False, "error": "Invalid source person name"}), 400

    data = request.get_json(silent=True) or {}
    target_name = safe_person_name(data.get("target", ""))
    if not target_name:
        return jsonify({"ok": False, "error": "target person name is required"}), 400
    if person == target_name:
        return jsonify({"ok": False, "error": "Source and target are the same"}), 400

    filenames = data.get("filenames") or []
    if not isinstance(filenames, list) or not filenames:
        return jsonify({"ok": False, "error": "filenames list is required"}), 400

    src_dir = os.path.join(FACES_DIR, person)
    if not os.path.isdir(src_dir):
        return jsonify({"ok": False, "error": "Source person not found"}), 404

    tgt_dir = os.path.join(FACES_DIR, target_name)
    os.makedirs(tgt_dir, exist_ok=True)

    moved, missing = [], []
    for raw in filenames:
        fn = secure_filename(str(raw))
        if not fn:
            missing.append(raw); continue
        sp = os.path.join(src_dir, fn)
        if not os.path.isfile(sp):
            missing.append(fn); continue
        ext = os.path.splitext(fn)[1].lower() or ".jpg"
        new_fn = next_image_filename(tgt_dir, ext)
        try:
            shutil.move(sp, os.path.join(tgt_dir, new_fn))
            moved.append({"old": fn, "new": new_fn})
        except Exception:
            missing.append(fn)

    person_removed = False
    if os.path.isdir(src_dir):
        remaining = [f for f in os.listdir(src_dir) if os.path.splitext(f)[1].lower() in ALLOWED_EXTS]
        if not remaining:
            shutil.rmtree(src_dir, ignore_errors=True)
            person_removed = True

    _reload_faces_async()
    return jsonify({
        "ok": True, "target": target_name, "moved": moved,
        "missing": missing, "person_removed": person_removed,
    })


@app.route("/api/people/merge", methods=["POST"])
def api_merge_people():
    """Merge one or more source people into a target person.

    Body: {"sources": ["a", "b"], "target": "c"}. The target may be a new name —
    it will be created. All images from each source are moved into the target's
    folder (filenames re-numbered to avoid collisions). Source folders are
    removed when emptied.
    """
    data = request.get_json(silent=True) or {}
    target_name = safe_person_name(data.get("target", ""))
    sources_raw = data.get("sources") or []
    if not target_name:
        return jsonify({"ok": False, "error": "target is required"}), 400
    if not isinstance(sources_raw, list) or not sources_raw:
        return jsonify({"ok": False, "error": "sources list is required"}), 400

    sources = []
    for s in sources_raw:
        sn = safe_person_name(s)
        if sn and sn != target_name and sn not in sources:
            sources.append(sn)
    if not sources:
        return jsonify({"ok": False, "error": "no valid sources"}), 400

    tgt_dir = os.path.join(FACES_DIR, target_name)
    os.makedirs(tgt_dir, exist_ok=True)

    merged, removed = 0, []
    for src in sources:
        src_dir = os.path.join(FACES_DIR, src)
        if not os.path.isdir(src_dir):
            continue
        for fn in list(os.listdir(src_dir)):
            ext = os.path.splitext(fn)[1].lower()
            if ext not in ALLOWED_EXTS:
                continue
            sp = os.path.join(src_dir, fn)
            new_fn = next_image_filename(tgt_dir, ext or ".jpg")
            try:
                shutil.move(sp, os.path.join(tgt_dir, new_fn))
                merged += 1
            except Exception:
                pass
        try:
            remaining = [f for f in os.listdir(src_dir) if os.path.splitext(f)[1].lower() in ALLOWED_EXTS]
            if not remaining:
                shutil.rmtree(src_dir, ignore_errors=True)
                removed.append(src)
        except Exception:
            pass

    # Reassign visit history rows from each source to the target so that
    # historical visits appear under the merged identity.
    rows_updated = 0
    if db.is_available():
        for src in sources:
            try:
                rows_updated += int(db.rename_person(src, target_name) or 0)
            except Exception:
                pass

    _reload_faces_async()
    return jsonify({
        "ok": True, "target": target_name, "sources": sources,
        "merged": merged, "removed": removed,
        "visits_reassigned": rows_updated,
    })


_STATIC_CACHE_HEADER = "public, max-age=86400"


@app.route("/faces/<person>/<path:filename>")
def faces_file(person, filename):
    # prevent path traversal / weird names
    person = safe_person_name(person)
    if not person:
        return ("", 404)

    resp = send_from_directory(os.path.join(FACES_DIR, person), filename, conditional=True)
    resp.headers["Cache-Control"] = _STATIC_CACHE_HEADER
    return resp

@app.route("/footage/<path:filename>")
def footage_file(filename):
    """Serve visit footage video clips (supports Range requests for streaming)."""
    resp = send_from_directory(FOOTAGE_DIR, filename, conditional=True)
    resp.headers["Cache-Control"] = _STATIC_CACHE_HEADER
    return resp

@app.route("/api/status")
def api_status():
    return jsonify({
        "running": engine.is_running(),
        "identities": len(engine.known_embeddings),
        "camera_source": _camera_source_to_text(engine.cam_index),
        "action_enabled": engine.activity_enabled,
    })


@app.route("/api/camera", methods=["GET"])
def api_camera_get():
    # Determine which camera indices the engine currently holds open
    active_cams = []
    if engine.is_running():
        if engine._is_grid_mode():
            sources = engine._parse_grid_sources(engine.cam_index)
            active_cams = [s for s in sources if isinstance(s, int)]
        elif isinstance(engine.cam_index, int):
            active_cams = [engine.cam_index]
    rows, cols = engine._grid_layout
    loading_opened = int(_engine_state.get("loading_opened", 0) if _engine_state else 0)
    loading_total = int(_engine_state.get("loading_total", 0) if _engine_state else 0)
    return jsonify(
        {
            "running": engine.is_running(),
            "camera_source": _camera_source_to_text(engine.cam_index),
            "devices": _list_camera_devices(),
            "active_cameras": active_cams,
            "viewer_mode": engine.viewer_mode,
            "viewer_source": engine.viewer_source,
            "viewer_grid_offset": engine.viewer_grid_offset,
            "grid_layout": [rows, cols],
            "grid_page_size": engine.grid_page_size(),
            "grid_page_count": engine.grid_page_count(),
            "loading_opened": loading_opened,
            "loading_total": loading_total,
        }
    )


def _all_configured_sources():
    """Return every non-grid camera source value the UI knows about."""
    out = []
    for d in _list_camera_devices():
        val = str(d.get("value", "")).strip()
        if not val or val.startswith("grid_"):
            continue
        out.append(val)
    return out


def _build_analysis_pool_source():
    """Build the engine's grid: cam_index string covering every configured
    camera. Empty if there are none."""
    sources = _all_configured_sources()
    if not sources:
        return ""
    return "grid:" + ",".join(sources)


def _refresh_source_name_map():
    """Rebuild engine.source_name_map from the IP-cameras config so grid
    tiles label IP cams by their configured name."""
    engine.source_name_map = {
        url: (cam.get("name") or "IP Camera")
        for _, cam, url in _expanded_cameras(_load_ip_cameras())
    }


@app.route("/api/camera", methods=["POST"])
def api_camera_set():
    """Update *viewer* state. Does not stop or restart the engine —
    every configured camera stays open and continues running face
    recognition. Choosing a grid layout (``grid_RxC``) just changes
    the visible composite size; choosing a numeric or URL source
    switches single-view to that camera.
    """
    payload = request.get_json(silent=True) or {}
    # Support a "grid_offset" only update — change which page of cameras
    # the visible composite shows without touching mode or layout.
    if "grid_offset" in payload and "source" not in payload:
        try:
            new_offset = int(payload.get("grid_offset", 0))
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "grid_offset must be int"}), 400
        # Wrap or clamp so the offset is always valid for the current pool
        page_size = engine.grid_page_size()
        n_pages = engine.grid_page_count()
        if n_pages > 0:
            page_idx = (new_offset // page_size) % n_pages
            new_offset = page_idx * page_size
        engine.set_viewer(grid_offset=new_offset)
        return jsonify({
            "ok": True,
            "running": engine.is_running(),
            "viewer_mode": engine.viewer_mode,
            "viewer_grid_offset": engine.viewer_grid_offset,
            "grid_page_count": n_pages,
        })

    source_raw = payload.get("source", request.form.get("source", ""))
    source_text = str(source_raw or "").strip()
    if not source_text:
        return jsonify({"ok": False, "error": "source is required"}), 400

    grid_match = re.fullmatch(r"grid_(\d+)x(\d+)", source_text)
    if grid_match:
        rows, cols = int(grid_match.group(1)), int(grid_match.group(2))
        old_layout = tuple(engine._grid_layout)
        try:
            engine.set_grid_layout(rows, cols)
        except ValueError as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        # Reset to first page when entering grid mode or changing layout
        engine.set_viewer(mode="grid", source="", grid_offset=0)
        # Layout actually changed → restart the analysis pool. The render
        # loop captures tile sizes at start; without restart, the new
        # layout would just paint into the old canvas dimensions and the
        # composite would still look like the previous grid.
        if (rows, cols) != old_layout and engine.is_running():
            with state_lock:
                if engine.is_running():
                    engine.stop()
                pool = _build_analysis_pool_source()
                if pool:
                    engine.cam_index = pool
                _refresh_source_name_map()
                engine.set_viewer(mode="grid", source="", grid_offset=0)
                try:
                    engine.start()
                except Exception as e:
                    return jsonify({"ok": False, "error": f"failed to restart: {e}"}), 500
    else:
        engine.set_viewer(mode="single", source=source_text)

    _refresh_source_name_map()

    rows_o, cols_o = engine._grid_layout
    # Read viewer_source directly from source_text rather than the shared
    # state dict — the mirror loop (50 ms tick) may not have synced yet,
    # which would return the previous camera and make the label lag one click.
    resolved_source = source_text if not grid_match else ""
    resolved_mode = "grid" if grid_match else "single"
    return jsonify({
        "ok": True,
        "running": engine.is_running(),
        "viewer_mode": resolved_mode,
        "viewer_source": resolved_source,
        "viewer_grid_offset": engine.viewer_grid_offset,
        "grid_layout": [rows_o, cols_o],
        "grid_page_size": engine.grid_page_size(),
        "grid_page_count": engine.grid_page_count(),
        "camera_source": _camera_source_to_text(engine.cam_index),
        "devices": _list_camera_devices(),
    })


@app.route("/api/camera/reload", methods=["POST"])
def api_camera_reload():
    """Restart the analysis pool to pick up newly added/removed cameras.
    The viewer state is preserved across the restart."""
    with state_lock:
        was_running = engine.is_running()
        prev_mode = engine.viewer_mode
        prev_source = engine.viewer_source
        if was_running:
            engine.stop()
        new_pool = _build_analysis_pool_source()
        if not new_pool:
            return jsonify({"ok": False, "error": "no cameras configured"}), 400
        engine.cam_index = new_pool
        _refresh_source_name_map()
        engine.set_viewer(mode=prev_mode, source=prev_source)
        try:
            engine.start()
        except Exception as e:
            return jsonify({"ok": False, "error": f"failed to start: {e}"}), 500
    return jsonify({"ok": True, "running": engine.is_running()})

def _all_resolved_urls(state, exclude_camera_id=None):
    """Return a set of resolved URLs across all groups, optionally
    excluding a given camera id (used for duplicate-detection on update)."""
    out = set()
    for _, c, url in _expanded_cameras(state):
        if exclude_camera_id and c.get("id") == exclude_camera_id:
            continue
        out.add(url)
    return out


@app.route("/api/ip_cameras", methods=["GET"])
def api_ip_cameras_list():
    return jsonify({"ok": True, **_serialize_state(_load_ip_cameras())})


@app.route("/api/ip_cameras/groups", methods=["POST"])
def api_ip_cameras_group_add():
    payload = request.get_json(silent=True) or {}
    name = str(payload.get("name", "")).strip() or "Group"
    base_url = str(payload.get("base_url", "")).strip()
    with _ip_cameras_lock:
        state = _load_ip_cameras()
        group = {"id": _new_id(), "name": name, "base_url": base_url, "cameras": []}
        state["groups"].append(group)
        _save_ip_cameras(state)
    return jsonify({"ok": True, "group": group})


@app.route("/api/ip_cameras/groups/<group_id>", methods=["PUT"])
def api_ip_cameras_group_update(group_id):
    payload = request.get_json(silent=True) or {}
    new_name = payload.get("name")
    new_base = payload.get("base_url")
    with _ip_cameras_lock:
        state = _load_ip_cameras()
        g = _find_group(state, group_id)
        if not g:
            return jsonify({"ok": False, "error": "group not found"}), 404
        if new_name is not None:
            n = str(new_name).strip()
            if n:
                g["name"] = n
        if new_base is not None:
            g["base_url"] = str(new_base).strip()
        _save_ip_cameras(state)
    return jsonify({"ok": True, "group": {
        "id": g["id"], "name": g["name"], "base_url": g["base_url"],
    }})


@app.route("/api/ip_cameras/groups/<group_id>", methods=["DELETE"])
def api_ip_cameras_group_delete(group_id):
    with _ip_cameras_lock:
        state = _load_ip_cameras()
        before = len(state["groups"])
        state["groups"] = [g for g in state["groups"] if g.get("id") != group_id]
        if len(state["groups"]) == before:
            return jsonify({"ok": False, "error": "group not found"}), 404
        _save_ip_cameras(state)
    return jsonify({"ok": True})


@app.route("/api/ip_cameras/groups/<group_id>/cameras", methods=["POST"])
def api_ip_cameras_camera_add(group_id):
    payload = request.get_json(silent=True) or {}
    name = str(payload.get("name", "")).strip() or "IP Camera"
    channel = str(payload.get("channel", "")).strip()
    url = str(payload.get("url", "")).strip()

    if not channel and not url:
        return jsonify({"ok": False, "error": "either channel or url is required"}), 400

    with _ip_cameras_lock:
        state = _load_ip_cameras()
        g = _find_group(state, group_id)
        if not g:
            return jsonify({"ok": False, "error": "group not found"}), 404

        cam = {"id": _new_id(), "name": name}
        if url:
            cam["url"] = url
        else:
            cam["channel"] = channel

        # Duplicate URL check (only if we can resolve a URL)
        resolved = _resolved_camera_url(cam, g.get("base_url", ""))
        if resolved and resolved in _all_resolved_urls(state):
            return jsonify({"ok": False, "error": "a camera with this URL already exists"}), 400

        g.setdefault("cameras", []).append(cam)
        _save_ip_cameras(state)
    return jsonify({"ok": True, "camera": {**cam, "resolved_url": resolved}})


@app.route("/api/ip_cameras/cameras/<camera_id>", methods=["PUT"])
def api_ip_cameras_camera_update(camera_id):
    payload = request.get_json(silent=True) or {}
    new_name = payload.get("name")
    new_channel = payload.get("channel")
    new_url = payload.get("url")
    with _ip_cameras_lock:
        state = _load_ip_cameras()
        group, cam = _find_camera(state, camera_id)
        if not cam:
            return jsonify({"ok": False, "error": "camera not found"}), 404

        if new_name is not None:
            n = str(new_name).strip()
            if n:
                cam["name"] = n

        # channel and url are mutually exclusive on a single camera record
        if new_url is not None:
            u = str(new_url).strip()
            if u:
                cam["url"] = u
                cam.pop("channel", None)
            else:
                cam.pop("url", None)
        if new_channel is not None:
            ch = str(new_channel).strip()
            if ch:
                cam["channel"] = ch
                cam.pop("url", None)
            else:
                cam.pop("channel", None)

        # Duplicate URL check across all groups (excluding this camera)
        resolved = _resolved_camera_url(cam, group.get("base_url", ""))
        if resolved and resolved in _all_resolved_urls(state, exclude_camera_id=camera_id):
            return jsonify({"ok": False, "error": "another camera with this URL already exists"}), 400

        _save_ip_cameras(state)
    return jsonify({"ok": True, "camera": {**cam, "resolved_url": resolved}})


def _coerce_probe_source(raw):
    """Convert a UI source value into something cv2.VideoCapture accepts."""
    s = str(raw).strip()
    return int(s) if re.fullmatch(r"\d+", s) else s


def _probe_camera_url(url, timeout_s=8.0):
    """Try to open *url* with cv2.VideoCapture and read one frame.

    Accepts either a URL string or an integer-like device index. Returns
    ``(ok, message)``. Runs cv2 calls in a worker thread so we can enforce
    *timeout_s* even if the FFmpeg/V4L backend hangs.
    """
    src = _coerce_probe_source(url)
    is_url = isinstance(src, str)
    result = {"ok": False, "msg": "timeout"}

    def _worker():
        # Use PyAV for probing — cv2.VideoCapture can fail on H.265/HEVC
        # streams even when the camera is perfectly reachable.
        if is_url:
            try:
                import av as _av
                c = _av.open(src, options={"rtsp_transport": "tcp", "timeout": "5000000"})
                vs = next((s for s in c.streams if s.type == "video"), None)
                if vs:
                    w = vs.codec_context.width
                    h = vs.codec_context.height
                    codec = vs.codec_context.name
                    result["ok"] = True
                    result["msg"] = f"ok ({w}x{h} {codec})" if w and h else "ok"
                else:
                    result["ok"] = True
                    result["msg"] = "ok (no video stream info)"
                c.close()
                return
            except Exception as e:
                result["msg"] = str(e)
                return
        cap = None
        try:
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                result["msg"] = "device unavailable"
                return
            ok, frame = cap.read()
            if not ok or frame is None:
                result["msg"] = "opened but no frame received"
                return
            result["ok"] = True
            result["msg"] = f"ok ({frame.shape[1]}x{frame.shape[0]})"
        except Exception as e:
            result["msg"] = f"error: {e}"
        finally:
            if cap is not None:
                try: cap.release()
                except Exception: pass

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        return False, f"timed out after {timeout_s:.0f}s"
    return result["ok"], result["msg"]


@app.route("/api/ip_cameras/cameras/<camera_id>/test", methods=["POST"])
def api_ip_cameras_camera_test(camera_id):
    payload = request.get_json(silent=True) or {}
    url = str(payload.get("url", "")).strip()
    if not url:
        state = _load_ip_cameras()
        group, cam = _find_camera(state, camera_id)
        if not cam:
            return jsonify({"ok": False, "error": "camera not found"}), 404
        url = _resolved_camera_url(cam, group.get("base_url", ""))
    if not url:
        return jsonify({"ok": False, "error": "camera has no resolvable url"}), 400
    ok, msg = _probe_camera_url(url)
    return jsonify({"ok": ok, "message": msg})


@app.route("/api/ip_cameras/cameras/<camera_id>", methods=["DELETE"])
def api_ip_cameras_camera_delete(camera_id):
    with _ip_cameras_lock:
        state = _load_ip_cameras()
        group, cam = _find_camera(state, camera_id)
        if not cam:
            return jsonify({"ok": False, "error": "camera not found"}), 404
        group["cameras"] = [c for c in group["cameras"] if c.get("id") != camera_id]
        _save_ip_cameras(state)
    return jsonify({"ok": True})


@app.route("/api/ip_cameras/groups/<group_id>/reorder", methods=["POST"])
def api_ip_cameras_group_reorder(group_id):
    """Reorder cameras within a group. Body: {"order": ["cam_id", ...]}.

    Cameras whose IDs aren't in the list are kept in their existing
    relative order, appended after the supplied ones. Unknown IDs are
    ignored.
    """
    data = request.get_json(silent=True) or {}
    order = data.get("order") or []
    if not isinstance(order, list):
        return jsonify({"ok": False, "error": "order must be a list of camera ids"}), 400

    with _ip_cameras_lock:
        state = _load_ip_cameras()
        group = next((g for g in state.get("groups", []) if g.get("id") == group_id), None)
        if not group:
            return jsonify({"ok": False, "error": "group not found"}), 404

        cams_by_id = {c.get("id"): c for c in group.get("cameras", [])}
        seen = set()
        new_list = []
        for cid in order:
            cam = cams_by_id.get(cid)
            if cam and cid not in seen:
                new_list.append(cam)
                seen.add(cid)
        # Append any cameras the client didn't include, preserving order.
        for cam in group.get("cameras", []):
            if cam.get("id") not in seen:
                new_list.append(cam)
        group["cameras"] = new_list
        _save_ip_cameras(state)

    return jsonify({"ok": True, "count": len(new_list)})


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
    return render_template("settings.html")


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
        "location_display": _resolve_camera_display_name(v.get("camera_source")),
        "camera_source": v.get("camera_source", ""),
        "first_seen": first.isoformat() if first else None,
        "last_seen": last.isoformat() if last else None,
        "duration_secs": round(duration_secs, 1),
        "duration_fmt": _fmt_duration(duration_secs),
        "ended": bool(v.get("ended", False)),
        "confidence": round(float(v["confidence"]), 3) if v.get("confidence") else None,
        "footage_url": f"/footage/{v['footage']}" if v.get("footage") else None,
        "activity": v.get("activity"),
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
    for l in locs:
        l["display_name"] = _resolve_camera_display_name(l.get("camera_source"))
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
            # Always start with the full analysis pool — every USB and IP
            # camera the system knows about. The viewer state decides what
            # the MJPEG stream displays; analysis runs on all of them.
            pool = _build_analysis_pool_source()
            if pool:
                engine.cam_index = pool
            _refresh_source_name_map()
            try:
                engine.start()
            except Exception as e:
                return jsonify({"ok": False, "error": f"failed to start: {e}"}), 500
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


@app.route("/api/analytics/summary")
def api_analytics_summary():
    """Single-request summary tiles for the analytics dashboard.

    Query params:
      ?date=YYYY-MM-DD  (default: today in local time)

    Returns:
      peak_hour        str   e.g. "09:00 – 10:00" (local time), or null if no data
      unknowns_today   int   distinct unknown_N persons with at least one visit today
      present_today    int   distinct known persons with at least one visit today
    """
    if not db.is_available():
        return jsonify({"ok": False, "error": "database not available"}), 503

    date_str = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    try:
        local_midnight = datetime.strptime(date_str, "%Y-%m-%d").astimezone()
    except ValueError:
        return jsonify({"ok": False, "error": "invalid date format"}), 400

    day_start = local_midnight.astimezone(timezone.utc)
    day_end   = (local_midnight + timedelta(days=1)).astimezone(timezone.utc)
    ph = "?" if db._backend == "sqlite" else "%s"

    ds = day_start.isoformat() if db._backend == "sqlite" else day_start
    de = day_end.isoformat()   if db._backend == "sqlite" else day_end

    # Peak hour — hour bucket (local time) with the most visit records
    if db._backend == "sqlite":
        hour_sql = f"""
            SELECT strftime('%H', datetime(first_seen, 'localtime')) as hr,
                   COUNT(DISTINCT person_name) as cnt
            FROM visits
            WHERE first_seen >= {ph} AND first_seen < {ph}
            GROUP BY hr ORDER BY cnt DESC LIMIT 1
        """
    else:
        hour_sql = f"""
            SELECT date_part('hour', first_seen AT TIME ZONE 'localtime') as hr,
                   COUNT(DISTINCT person_name) as cnt
            FROM visits
            WHERE first_seen >= {ph} AND first_seen < {ph}
            GROUP BY hr ORDER BY cnt DESC LIMIT 1
        """

    # People present today — distinct known persons seen today
    present_sql = f"""
        SELECT COUNT(DISTINCT person_name) as cnt FROM visits
        WHERE first_seen >= {ph} AND first_seen < {ph}
          AND person_name NOT LIKE 'unknown_%'
    """

    params = (ds, de)
    with db._cursor() as cur:
        cur.execute(hour_sql, params)
        hour_row = db._row_to_dict(cur.fetchone())

        cur.execute(present_sql, params)
        present_row = db._row_to_dict(cur.fetchone())

    # Count named faces/ folders (excludes unknown_N) for absent calculation
    known_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faces")
    import re as _re
    unknowns_count = 0
    enrolled_known = 0
    if os.path.isdir(known_dir):
        for d in os.listdir(known_dir):
            if not os.path.isdir(os.path.join(known_dir, d)):
                continue
            if _re.match(r'^unknown_\d+$', d):
                unknowns_count += 1
            else:
                enrolled_known += 1

    present_today = int(present_row["cnt"]) if present_row else 0
    absent_today  = max(0, enrolled_known - present_today)

    peak_hour = None
    if hour_row and hour_row.get("hr") is not None:
        h = int(hour_row["hr"])
        peak_hour = f"{h:02d}:00 – {(h + 1) % 24:02d}:00"

    return jsonify({
        "ok": True,
        "date": date_str,
        "peak_hour":      peak_hour,
        "present_today":  present_today,
        "absent_today":   absent_today,
        "unknowns_today": unknowns_count,
    })


@app.route("/api/analytics/present_absent")
def api_analytics_present_absent():
    """Return lists of present and absent known persons for a given day.

    Query params:
      ?date=YYYY-MM-DD  (default: today in local time)

    Returns:
      present  list of person names with at least one visit today
      absent   list of enrolled known persons with no visit today
    """
    if not db.is_available():
        return jsonify({"ok": False, "error": "database not available"}), 503

    date_str = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    try:
        local_midnight = datetime.strptime(date_str, "%Y-%m-%d").astimezone()
    except ValueError:
        return jsonify({"ok": False, "error": "invalid date format"}), 400

    day_start = local_midnight.astimezone(timezone.utc)
    day_end   = (local_midnight + timedelta(days=1)).astimezone(timezone.utc)
    ph = "?" if db._backend == "sqlite" else "%s"
    ds = day_start.isoformat() if db._backend == "sqlite" else day_start
    de = day_end.isoformat()   if db._backend == "sqlite" else day_end

    present_sql = f"""
        SELECT DISTINCT person_name FROM visits
        WHERE first_seen >= {ph} AND first_seen < {ph}
          AND person_name NOT LIKE 'unknown_%'
        ORDER BY person_name
    """
    with db._cursor() as cur:
        cur.execute(present_sql, (ds, de))
        present = [r["person_name"] for r in db._rows_to_dicts(cur.fetchall())]

    # Absent = enrolled known folders not in present list
    known_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faces")
    import re as _re
    absent = []
    if os.path.isdir(known_dir):
        present_set = set(present)
        for d in sorted(os.listdir(known_dir)):
            if not os.path.isdir(os.path.join(known_dir, d)):
                continue
            if _re.match(r'^unknown_\d+$', d):
                continue
            if d not in present_set:
                absent.append(d)

    return jsonify({"ok": True, "date": date_str, "present": present, "absent": absent})


@app.route("/api/analytics/earliest")
def api_analytics_earliest():
    """Top 10 employees with the earliest first arrival on a specific date.

    Query params:
      ?date=YYYY-MM-DD  (default: today)
      ?order=latest     (reverse sort for latest arrivals)
      ?shift=morning|night
          morning = 04:00–16:00 on the given date
          night   = 16:00 on the given date to 04:00 the following day
          (omit for full day)
    Returns rows sorted by first_seen, excluding unknown_N names.
    """
    if not db.is_available():
        return jsonify({"ok": False, "error": "database not available"}), 503

    date_str = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    try:
        # Build shift boundaries in local time, then convert to UTC for the SQL comparison
        local_midnight = datetime.strptime(date_str, "%Y-%m-%d").astimezone()
        day = local_midnight
    except ValueError:
        return jsonify({"ok": False, "error": "invalid date format"}), 400

    shift = request.args.get("shift", "")  # "morning", "night", or ""
    order = "DESC" if request.args.get("order") == "latest" else "ASC"
    ph = "?" if db._backend == "sqlite" else "%s"

    if shift == "morning":
        start = (day + timedelta(hours=4)).astimezone(timezone.utc)
        end   = (day + timedelta(hours=16)).astimezone(timezone.utc)
    elif shift == "night":
        start = (day + timedelta(hours=16)).astimezone(timezone.utc)
        end   = (day + timedelta(hours=28)).astimezone(timezone.utc)  # 04:00 next day
    else:
        start = day.astimezone(timezone.utc)
        end   = (day + timedelta(days=1)).astimezone(timezone.utc)

    params = (start.isoformat() if db._backend == "sqlite" else start,
              end.isoformat()   if db._backend == "sqlite" else end)

    # For night shift: exclude anyone who already arrived during morning shift
    # so each person appears in at most one shift row.
    morning_exclusion = ""
    morning_params = ()
    if shift == "night":
        morning_start = (day + timedelta(hours=4)).astimezone(timezone.utc)
        morning_end   = (day + timedelta(hours=16)).astimezone(timezone.utc)
        ms = morning_start.isoformat() if db._backend == "sqlite" else morning_start
        me = morning_end.isoformat()   if db._backend == "sqlite" else morning_end
        morning_exclusion = f"""
          AND person_name NOT IN (
            SELECT DISTINCT person_name FROM visits
            WHERE first_seen >= {ph} AND first_seen < {ph}
          )"""
        morning_params = (ms, me)

    sql = f"""
        SELECT person_name, MIN(first_seen) as earliest
        FROM visits
        WHERE first_seen >= {ph} AND first_seen < {ph}
          AND person_name NOT LIKE 'unknown_%'
          {morning_exclusion}
        GROUP BY person_name
        ORDER BY earliest {order}
        LIMIT 10
    """
    with db._cursor() as cur:
        cur.execute(sql, params + morning_params)
        rows = db._rows_to_dicts(cur.fetchall())

    result = []
    for row in rows:
        dt = _to_dt(row["earliest"])
        result.append({
            "person_name": row["person_name"],
            "arrival_time": dt.astimezone().strftime("%I:%M %p") if dt else "--",
        })

    return jsonify({"ok": True, "date": date_str, "shift": shift, "rows": result})


@app.route("/api/analytics/longest")
def api_analytics_longest():
    """Top 10 employees with the longest total visible duration for a period.

    Query param: ?period=day|week|month|year (default: day)
    Sums visible_duration (seconds on camera) per person, falls back to
    last_seen - first_seen for visits without visible_duration recorded.
    """
    if not db.is_available():
        return jsonify({"ok": False, "error": "database not available"}), 503

    period = request.args.get("period", "day")
    now = datetime.now(timezone.utc)

    if period == "day":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "week":
        # Calendar week: Sunday = day 0. weekday() returns Mon=0..Sun=6, so
        # days_since_sunday = (weekday + 1) % 7
        days_since_sunday = (now.weekday() + 1) % 7
        start = (now - timedelta(days=days_since_sunday)).replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "month":
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    elif period == "year":
        start = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    else:
        return jsonify({"ok": False, "error": "invalid period"}), 400

    ph = "?" if db._backend == "sqlite" else "%s"
    start_val = start.isoformat() if db._backend == "sqlite" else start

    # Use visible_duration when available, otherwise derive from timestamps.
    if db._backend == "sqlite":
        duration_expr = "SUM(COALESCE(visible_duration, (julianday(last_seen) - julianday(first_seen)) * 86400))"
    else:
        duration_expr = "SUM(COALESCE(visible_duration, EXTRACT(EPOCH FROM (last_seen - first_seen))))"

    sql = f"""
        SELECT person_name, {duration_expr} as total_secs
        FROM visits
        WHERE first_seen >= {ph}
          AND person_name NOT LIKE 'unknown_%'
        GROUP BY person_name
        ORDER BY total_secs DESC
        LIMIT 10
    """
    with db._cursor() as cur:
        cur.execute(sql, (start_val,))
        rows = db._rows_to_dicts(cur.fetchall())

    result = []
    for row in rows:
        secs = float(row["total_secs"] or 0)
        result.append({
            "person_name": row["person_name"],
            "total_secs": round(secs, 1),
            "duration_fmt": _fmt_duration(secs),
        })

    return jsonify({"ok": True, "period": period, "rows": result})


@app.route("/api/analytics/headcount")
def api_analytics_headcount():
    """Distinct people present per day for a date range.

    Query params: ?from=YYYY-MM-DD&to=YYYY-MM-DD (both default to current month).
    Returns [{date, count}] ordered by date ascending.
    """
    if not db.is_available():
        return jsonify({"ok": False, "error": "database not available"}), 503

    now = datetime.now(timezone.utc)
    default_from = now.replace(day=1).strftime("%Y-%m-%d")
    default_to = now.strftime("%Y-%m-%d")
    from_date = request.args.get("from", default_from)
    to_date = request.args.get("to", default_to)

    ph = "?" if db._backend == "sqlite" else "%s"
    if db._backend == "sqlite":
        date_expr = "DATE(first_seen)"
    else:
        date_expr = "first_seen::date"

    sql = f"""
        SELECT {date_expr} as day, COUNT(DISTINCT person_name) as count
        FROM visits
        WHERE {date_expr} >= {ph} AND {date_expr} <= {ph}
          AND person_name NOT LIKE 'unknown_%'
        GROUP BY day
        ORDER BY day ASC
    """
    with db._cursor() as cur:
        cur.execute(sql, (from_date, to_date))
        rows = db._rows_to_dicts(cur.fetchall())

    return jsonify({"ok": True, "rows": [{"date": r["day"], "count": int(r["count"])} for r in rows]})


@app.route("/api/analytics/heatmap")
def api_analytics_heatmap():
    """Presence heatmap: which employees were present on which days.

    Query params: ?from=YYYY-MM-DD&to=YYYY-MM-DD (both default to current month).
    Returns {dates, persons, present: {person: {date: true}}}.
    """
    if not db.is_available():
        return jsonify({"ok": False, "error": "database not available"}), 503

    now = datetime.now(timezone.utc)
    default_from = now.replace(day=1).strftime("%Y-%m-%d")
    default_to = now.strftime("%Y-%m-%d")
    from_date = request.args.get("from", default_from)
    to_date = request.args.get("to", default_to)

    ph = "?" if db._backend == "sqlite" else "%s"
    if db._backend == "sqlite":
        date_expr = "DATE(first_seen)"
    else:
        date_expr = "first_seen::date"

    sql = f"""
        SELECT {date_expr} as day, person_name
        FROM visits
        WHERE {date_expr} >= {ph} AND {date_expr} <= {ph}
          AND person_name NOT LIKE 'unknown_%'
        GROUP BY day, person_name
        ORDER BY person_name ASC, day ASC
    """
    with db._cursor() as cur:
        cur.execute(sql, (from_date, to_date))
        rows = db._rows_to_dicts(cur.fetchall())

    dates_set = sorted({r["day"] for r in rows})
    persons_set = sorted({r["person_name"] for r in rows})
    present = {p: {} for p in persons_set}
    for r in rows:
        present[r["person_name"]][r["day"]] = True

    return jsonify({"ok": True, "dates": dates_set, "persons": persons_set, "present": present})


if __name__ == "__main__":
    # Spawn the engine subprocess and run engine-dependent boot steps
    # (DB session, background loops). Doing this here, after the
    # `if __name__ == "__main__"` guard, prevents Python's spawn start
    # method from recursing when the child re-imports this module.
    _bootstrap_engine()
    _bootstrap_post_engine()

    # The Werkzeug dev server stalls under load: long-lived MJPEG / SSE
    # streams hog its workers, so polling endpoints time out under load.
    # Use waitress (pure-Python production server) when available — it
    # handles many concurrent connections cleanly. Never enable the
    # Flask reloader: it would start the engine subprocess twice.
    host, port = "0.0.0.0", 5000
    try:
        from waitress import serve  # type: ignore
        log.info("Serving with waitress on %s:%s", host, port)
        serve(app, host=host, port=port, threads=32, channel_timeout=300)
    except ImportError:
        log.warning("waitress not installed — falling back to Werkzeug dev server. "
                    "pip install waitress for production-grade concurrency.")
        app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)
