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
os.environ.setdefault("INFERENCE_POOL_SIZE", "3")
os.environ.setdefault("MAX_CONCURRENT_RTSP", "20")
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
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
_LOG_FMT   = "[web]    %(asctime)s %(levelname)s %(message)s"

logging.basicConfig(level=_LOG_LEVEL, format=_LOG_FMT)

# File handler — logs/app.log, rotating at 5 MB, keeping 5 backups
from logging.handlers import RotatingFileHandler as _RFH
os.makedirs("logs", exist_ok=True)
_fh = _RFH("logs/app.log", maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
_fh.setFormatter(logging.Formatter(_LOG_FMT))
_fh.setLevel(_LOG_LEVEL)
logging.getLogger().addHandler(_fh)

# Suppress "Client disconnected while serving …" noise — these fire every
# time a browser closes an SSE/MJPEG/footage stream, which is normal.
logging.getLogger("werkzeug").addFilter(
    lambda r: "Client disconnected" not in r.getMessage()
)
logging.getLogger("waitress").addFilter(
    lambda r: "Client disconnected" not in r.getMessage()
        and "Task queue depth" not in r.getMessage()
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
IP_CAMERAS_PATH       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ip_cameras.json")
TRACKER_CONFIG_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tracker_config.json")
TRACKER_SNAPSHOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "tracker_snapshots")
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
            "branch": str(g.get("branch") or "Riyadh").strip(),
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


def _load_tracker_config():
    try:
        with open(TRACKER_CONFIG_PATH) as f:
            d = json.load(f)
        raw_transforms = d.get("cam_transforms") or {}
        cam_transforms = {}
        for src, t in raw_transforms.items():
            if not isinstance(t, dict):
                continue
            cam_transforms[str(src)] = {
                "zoom":   max(0.5, min(8.0, float(t.get("zoom", 1.0)))),
                "panX":   float(t.get("panX", 0.0)),
                "panY":   float(t.get("panY", 0.0)),
                "rotate": int(t.get("rotate", 0)) % 360,
            }
        raw_rois = d.get("tracker_rois") or {}
        tracker_rois = {}
        for src, r in raw_rois.items():
            if isinstance(r, (list, tuple)) and len(r) == 4:
                tracker_rois[str(src)] = [max(0.0, min(1.0, float(v))) for v in r]
        return {
            "cameras":        [str(s) for s in (d.get("cameras") or [])][:4],
            "line_y_ratio":   float(d.get("line_y_ratio", 0.5)),
            "line_y_ratios":  {str(k): max(0.05, min(0.95, float(v)))
                               for k, v in (d.get("line_y_ratios") or {}).items()},
            "cam_transforms": cam_transforms,
            "tracker_rois":   tracker_rois,
        }
    except Exception:
        return {"cameras": [], "line_y_ratio": 0.5, "line_y_ratios": {}, "cam_transforms": {}, "tracker_rois": {}}


def _save_tracker_config(config):
    with open(TRACKER_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


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


def _get_camera_branch(camera_source):
    """Return the branch name for a camera_source URL by scanning ip_cameras.json.

    Checks which group's resolved URLs contain camera_source and returns that
    group's branch. Defaults to 'Riyadh' if not found.
    """
    try:
        state = _load_ip_cameras()
        for g, c, url in _expanded_cameras(state):
            if str(url) == str(camera_source):
                return g.get("branch") or "Riyadh"
    except Exception:
        pass
    return "Riyadh"


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
            "branch": g.get("branch") or "Riyadh",
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
    threshold=0.6,
    detect_every=10,
    detect_scale=float(os.getenv("DETECT_SCALE", "0.5")),
    tracker_type="CSRT",
    width=1280,
    height=720,
    out_fps=15,
    jpeg_quality=85,
    # Draw bounding boxes + name labels on the live MJPEG feed.
    # Default on. Set LIVE_ANNOTATIONS_ENABLED=0 in .env to get a clean
    # stream — useful when many people are in frame and box overlays
    # become illegible. Footage recordings and per-visit screenshots
    # are always annotated regardless of this flag.
    live_annotations=_env_bool("LIVE_ANNOTATIONS_ENABLED", True),
    min_face_size=int(os.getenv("MIN_FACE_SIZE", "10")),
    tracker_snapshots_dir=TRACKER_SNAPSHOTS_DIR,
)

# Module-level placeholder so route handlers (which only resolve `engine`
# at request time) can be defined before bootstrap runs.
engine = None
_engine_manager = None
_engine_state = None
_engine_proc = None
_engine_parent_conn = None
_engine_shutting_down = False


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
        global _engine_shutting_down
        _engine_shutting_down = True
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
        if _engine_shutting_down:
            return
        try:
            if _engine_proc is not None and not _engine_proc.is_alive():
                exit_code = _engine_proc.exitcode
                log.warning("engine subprocess died (exit code %s) — restarting", exit_code)
                try: _engine_proc.join(timeout=2)
                except Exception: pass

                # Clear stale grid config so the new process starts fresh
                # instead of loading a potentially corrupted saved list.
                try:
                    _grid_cfg_path = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)), "grid_config.json",
                    )
                    if os.path.isfile(_grid_cfg_path):
                        os.remove(_grid_cfg_path)
                except Exception:
                    pass

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
                            _cam_urls = sorted(set(
                                url for _, _, url in _expanded_cameras(_load_ip_cameras())
                            ))
                            if _cam_urls:
                                engine.set_grid_layout(2, 2)
                                engine.set_viewer(mode="single", source=_cam_urls[0])
                            engine.start()
                            log.info("engine watchdog: auto-resumed camera grid")
                except Exception:
                    log.exception("engine watchdog: auto-resume failed")
        except Exception:
            log.exception("engine watchdog error")

state_lock = threading.Lock()

# Attendance state (in-memory, per server run)
attendance_lock = threading.Lock()
attendance_state = {}  # name -> {attended: bool, first_seen_ts: float, last_seen_mono: float, present: bool}
attendance_events_q    = Queue(maxsize=200)
tracker_events_q       = Queue(maxsize=500)
_tracker_active_t      = 0.0   # monotonic timestamp of last tracker ping
_TRACKER_ACTIVE_TTL    = 8.0   # seconds; tracker considered active if pinged within this window
ATTENDANCE_LOOP_SECS = 0.3
ATTENDANCE_DISAPPEAR_SECS = 2.0
UNKNOWN_PROMPT_SECS = 5.0
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
    threading.Thread(target=_daily_report_scheduler_loop, daemon=True, name="daily-report-scheduler").start()
    threading.Thread(target=_tracker_poll_loop, daemon=True, name="tracker-poll").start()
    try:
        _rc = _load_reports_config()
        _hp = [s for s in [_rc.get("arrival_camera", ""), _rc.get("exit_camera", "")] if s]
        if _hp:
            engine.set_config({"high_priority_sources": _hp})
            log.info("High-priority gate cameras registered: %s", _hp)
    except Exception as _e:
        log.warning("Could not set high_priority_sources: %s", _e)
    try:
        _tc = _load_tracker_config()
        if _tc["cameras"]:
            engine.set_tracker_cameras(_tc["cameras"])
            engine.set_config({"tracker_line_y_ratios": _tc.get("line_y_ratios", {}),
                               "tracker_rois": _tc.get("tracker_rois", {})})
            log.info("Tracker cameras configured: %s", _tc["cameras"])
    except Exception as _e:
        log.warning("tracker camera init: %s", _e)

    # Auto-start: cycle through all IP cameras every 3 minutes
    try:
        _all_camera_urls = sorted(set(
            url for _, _, url in _expanded_cameras(_load_ip_cameras())
        ))
        if not _all_camera_urls:
            pool = _build_analysis_pool_source()
            if pool:
                engine.cam_index = pool
                _refresh_source_name_map()
                engine.start()
                log.info("engine auto-started with pool: %s", pool)
        elif len(_all_camera_urls) == 1:
            _refresh_source_name_map()
            with state_lock:
                engine.cam_index = _all_camera_urls[0]
                engine.start()
            log.info("engine auto-started with single camera: %s", _all_camera_urls[0])
        else:
            _refresh_source_name_map()
            pool = "grid:" + ",".join(_all_camera_urls)
            with state_lock:
                engine.cam_index = pool
                engine.set_grid_layout(2, 2)
                engine.set_viewer(mode="single", source=_all_camera_urls[0])
                engine.start()
            log.info("engine auto-started: pool of %d cameras, viewer on first camera",
                     len(_all_camera_urls))
    except Exception as _e:
        log.warning("auto-start failed: %s", _e)

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
    BATCH_SIZE = 10
    while True:
        ops = []
        try:
            ops.append(_db_writer_q.get(timeout=0.1))
        except Empty:
            continue
        if ops[0] is None:
            return
        # Batch up to BATCH_SIZE ops without blocking for more than one.
        while len(ops) < BATCH_SIZE:
            try:
                ops.append(_db_writer_q.get_nowait())
            except Empty:
                break
        for op in ops:
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


def _tracker_poll_loop():
    """Poll the engine for tracker crossing events, persist to DB, and broadcast via SSE."""
    os.makedirs(TRACKER_SNAPSHOTS_DIR, exist_ok=True)
    while True:
        try:
            if engine is not None and engine.is_running():
                for ev in engine.pop_tracker_crossing_events():
                    cam_name = (engine.source_name_map or {}).get(
                        ev.get("camera_source", ""), ev.get("camera_source", "")
                    )
                    row_id = None
                    try:
                        row_id = db.insert_tracker_event(
                            event_type=ev["direction"],
                            person_name=ev.get("name", "unknown"),
                            camera_source=ev.get("camera_source", ""),
                            camera_name=cam_name,
                            snapshot_path=ev.get("snapshot_path"),
                            confidence=float(ev.get("best", 0.0)),
                        )
                    except Exception as _e:
                        log.warning("tracker DB insert: %s", _e)
                    payload = {
                        "id":           row_id,
                        "event_type":   ev.get("direction"),
                        "person_name":  ev.get("name", "unknown"),
                        "camera_name":  cam_name,
                        "snapshot_url": f"/{ev['snapshot_path']}" if ev.get("snapshot_path") else None,
                        "confidence":   float(ev.get("best", 0.0)),
                        "occurred_at":  datetime.now(timezone.utc).isoformat(),
                    }
                    try:
                        tracker_events_q.put_nowait({"event": "crossing", "data": payload})
                    except Exception:
                        pass
        except Exception as _e:
            log.debug("tracker_poll_loop: %s", _e)
        time.sleep(0.5)


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
        fname = f"visit_{visit_id}.mp4"
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


def _get_gate_cameras():
    """Return (arrival_camera, exit_camera) from config. Cached for 60s."""
    now = time.monotonic()
    if now - _gate_camera_cache["t"] > 60:
        cfg = _load_reports_config()
        _gate_camera_cache["arrival"] = cfg.get("arrival_camera", "")
        _gate_camera_cache["exit"] = cfg.get("exit_camera", "")
        _gate_camera_cache["t"] = now
    return _gate_camera_cache["arrival"], _gate_camera_cache["exit"]

_gate_camera_cache = {"arrival": "", "exit": "", "t": 0.0}


def _handle_gate_event(person_name, camera_source):
    """Called on every new visit / camera transition. Writes to gate_events."""
    if not db.is_available():
        return
    try:
        arrival_cam, exit_cam = _get_gate_cameras()
        now_dt = datetime.now(timezone.utc)
        if exit_cam and camera_source == exit_cam:
            db.open_gate_exit(person_name, now_dt)
        elif arrival_cam and camera_source == arrival_cam:
            # Try to close an open exit event (normal two-camera flow).
            closed = db.close_gate_entry(person_name, now_dt)
            if closed is None:
                # No open exit event — exit camera may be offline.
                # Write a standalone arrival record so the report still shows
                # arrival time even without exit-camera data.
                db.write_arrival_event(person_name, now_dt)
    except Exception as e:
        log.debug("gate_event error for %s on %s: %s", person_name, camera_source, e)


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
        branch = _get_camera_branch(camera_source)
        vid = db.open_visit(name, loc_id, confidence=confidence, session_id=_current_session_id, branch=branch)
        _active_visits[name] = {
            "location_id": loc_id,
            "visit_id": vid,
            "camera_source": camera_source,
            "last_seen_mono": now_mono,
        }
        _start_visit_footage(vid, name, camera_source)
        _handle_gate_event(name, camera_source)
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
            branch = _get_camera_branch(camera_source)
            vid = db.open_visit(name, loc_id, confidence=confidence, session_id=_current_session_id, branch=branch)
            _active_visits[name] = {
                "location_id": loc_id,
                "visit_id": vid,
                "camera_source": camera_source,
                "last_seen_mono": now_mono,
            }
            _start_visit_footage(vid, name, camera_source)
            _handle_gate_event(name, camera_source)

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

def _daily_report_scheduler_loop():
    """Background thread: save a report snapshot once per day at daily_send_time and email it."""
    import time as _time
    _last_saved_date = None
    while True:
        _time.sleep(30)
        try:
            if not db.is_available():
                continue
            cfg = _load_reports_config()
            arrival_cam = cfg.get("arrival_camera", "")
            exit_cam = cfg.get("exit_camera", "")
            if not arrival_cam and not exit_cam:
                continue

            now_local = datetime.now()
            today = now_local.strftime("%Y-%m-%d")
            send_hour = int(cfg.get("daily_send_time", "19:00").split(":")[0])
            due = now_local.hour >= send_hour

            if due and _last_saved_date != today:
                _last_saved_date = today
                try:
                    date_from = datetime.strptime(today, "%Y-%m-%d").astimezone(timezone.utc)
                    date_to = date_from + timedelta(days=1)
                    late_threshold = int(cfg.get("late_threshold_minutes", 15))
                    work_start = cfg.get("work_start", "08:00")
                    night_shift_enabled = bool(cfg.get("night_shift_enabled", False))
                    night_work_start = cfg.get("night_work_start", "21:00")
                    night_late_threshold = int(cfg.get("night_late_threshold_minutes", 15))
                    records = _generate_gate_report(arrival_cam, exit_cam, date_from, date_to, late_threshold, work_start,
                                                    night_shift_enabled, night_work_start, night_late_threshold)
                    db.save_daily_report(
                        today, arrival_cam, exit_cam, work_start,
                        late_threshold, records, sent_email=False,
                    )
                    log.info("Daily report saved for %s (%d records)", today, len(records))
                    # Send the email
                    ok, msg = _send_report_email(records, cfg, today, send_cc=True)
                    if ok:
                        log.info("Daily report emailed for %s", today)
                        db.save_daily_report(
                            today, arrival_cam, exit_cam, work_start,
                            late_threshold, records, sent_email=True,
                        )
                    else:
                        log.warning("Daily report email failed for %s: %s", today, msg)
                except Exception as e:
                    log.warning("Daily report save failed: %s", e)
        except Exception as e:
            log.warning("Daily report scheduler error: %s", e)

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
        devices.append({"value": url, "label": label, "path": url, "branch": group.get("branch") or "Riyadh"})

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
        # When the tracker page is active, throttle the composite stream to
        # 1 fps to free bandwidth for per-camera streams.
        if time.monotonic() - _tracker_active_t < _TRACKER_ACTIVE_TTL:
            time.sleep(1.0)
            continue

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
    try:
        engine.ping_viewer()
    except Exception:
        pass
    return Response(
        mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

@app.route("/video/<path:source>")
def video_camera(source):
    """Per-camera MJPEG stream. Loaded on-demand when the user navigates to a camera."""
    if not engine.is_running():
        return ("Camera stopped", 503)
    try:
        engine.ping_viewer()
    except Exception:
        pass

    def _gen():
        import numpy as _np
        import cv2 as _cv2

        def _make_no_signal_jpeg(label=""):
            h, w = 360, 640
            img = _np.zeros((h, w, 3), dtype=_np.uint8)
            _cv2.putText(img, "No Signal", (w // 2 - 120, h // 2 - 10),
                         _cv2.FONT_HERSHEY_SIMPLEX, 1.4, (80, 80, 80), 2)
            if label:
                short = label[-40:]
                _cv2.putText(img, short, (20, h // 2 + 40),
                             _cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
            ok, buf = _cv2.imencode(".jpg", img, [_cv2.IMWRITE_JPEG_QUALITY, 60])
            return bytes(buf) if ok else b""

        # Tracker cameras are pinned/high-priority — give them much longer before
        # showing "No Signal" so a brief reconnect doesn't flash the placeholder.
        _tracker_srcs = set(getattr(engine, "tracker_camera_sources", set()) or set())
        NO_SIGNAL_TIMEOUT = 30.0 if source in _tracker_srcs else 2.0
        interval = 1.0 / max(1, getattr(engine, "out_fps", 15))
        last_id = None
        last_frame_t = time.monotonic()
        no_signal_jpeg = None

        while engine.is_running():
            try:
                engine.ping_viewer()
            except Exception:
                pass
            frame = engine.get_camera_jpeg(source)
            if frame is None:
                # If no frame has arrived for NO_SIGNAL_TIMEOUT seconds, send a
                # placeholder so the browser replaces the stale previous-camera image.
                if time.monotonic() - last_frame_t > NO_SIGNAL_TIMEOUT:
                    if no_signal_jpeg is None:
                        no_signal_jpeg = _make_no_signal_jpeg(source)
                    if no_signal_jpeg:
                        try:
                            yield (
                                b"--frame\r\n"
                                b"Content-Type: image/jpeg\r\n\r\n"
                                + no_signal_jpeg
                                + b"\r\n"
                            )
                        except (GeneratorExit, BrokenPipeError, ConnectionResetError):
                            break
                    time.sleep(1.0)  # re-send placeholder at ~1 fps
                else:
                    time.sleep(0.05)
                continue
            last_frame_t = time.monotonic()
            no_signal_jpeg = None  # reset when real frames arrive again
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

    return Response(
        stream_with_context(_gen()),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

@app.route("/api/engine/config", methods=["GET", "POST"])
def engine_config():
    if not engine.is_running():
        return jsonify({"error": "Engine not running"}), 503
    if request.method == "POST":
        try:
            engine.set_config(request.json or {})
            return jsonify({"ok": True})
        except Exception as e:
            return jsonify({"error": str(e)}), 400
    return jsonify(engine.get_config())

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
    people = list_people()
    meta_map = {p["name"]: p for p in db.get_all_people_meta()}
    for p in people:
        m = meta_map.get(p["name"], {})
        p["section"]      = m.get("section", "")
        p["branch"]       = m.get("branch", "Riyadh")
        p["email"]        = m.get("email", "")
        p["arabic_name"]  = m.get("arabic_name", "")
        p["home_zone_id"] = m.get("home_zone_id")
    return jsonify({"people": people})


@app.route("/api/attendance/manual", methods=["POST"])
def api_manual_attendance():
    data = request.get_json(silent=True) or {}
    date_str   = (data.get("date")    or "").strip()
    arrived_str = (data.get("arrived") or "09:00").strip()
    left_str    = (data.get("left")    or "17:00").strip()
    names = data.get("names") or []
    if not date_str or not names:
        return jsonify({"ok": False, "error": "date and names are required"}), 400
    try:
        day = datetime.strptime(date_str, "%Y-%m-%d").date()
        arr_h, arr_m = map(int, arrived_str.split(":"))
        lft_h, lft_m = map(int, left_str.split(":"))
    except (ValueError, AttributeError):
        return jsonify({"ok": False, "error": "Invalid date or time format"}), 400
    first_seen = datetime(day.year, day.month, day.day, arr_h, arr_m).strftime("%Y-%m-%d %H:%M:%S")
    last_seen  = datetime(day.year, day.month, day.day, lft_h, lft_m).strftime("%Y-%m-%d %H:%M:%S")
    count = 0
    for name in names:
        name = str(name).strip()
        if not name:
            continue
        db.insert_manual_visit(name, first_seen, last_seen)
        count += 1
    return jsonify({"ok": True, "count": count})


@app.route("/api/sections", methods=["GET"])
def api_sections_list():
    sections = db.get_all_sections()
    for s in sections:
        s["members"] = db.get_section_members(s["name"])
    return jsonify({"ok": True, "sections": sections})


@app.route("/api/sections", methods=["POST"])
def api_sections_create():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"ok": False, "error": "Name is required"}), 400
    try:
        section = db.create_section(name)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    return jsonify({"ok": True, "section": section})


@app.route("/api/sections/<name>", methods=["DELETE"])
def api_sections_delete(name):
    db.delete_section(name)
    return jsonify({"ok": True})


@app.route("/api/sections/<name>/rename", methods=["POST"])
def api_sections_rename(name):
    data = request.get_json(silent=True) or {}
    new_name = (data.get("new_name") or "").strip()
    if not new_name:
        return jsonify({"ok": False, "error": "new_name is required"}), 400
    if new_name == name:
        return jsonify({"ok": True})
    db.rename_section(name, new_name)
    return jsonify({"ok": True, "new_name": new_name})


@app.route("/api/sections/<name>/assign", methods=["POST"])
def api_sections_assign(name):
    data = request.get_json(silent=True) or {}
    person = (data.get("person") or "").strip()
    if not person:
        return jsonify({"ok": False, "error": "person is required"}), 400
    db.assign_person_section(person, name)
    return jsonify({"ok": True})


@app.route("/api/sections/<name>/unassign", methods=["POST"])
def api_sections_unassign(name):
    data = request.get_json(silent=True) or {}
    person = (data.get("person") or "").strip()
    if not person:
        return jsonify({"ok": False, "error": "person is required"}), 400
    db.remove_person_section(person)
    # Clear manager if the unassigned person was the manager
    sections = db.get_all_sections()
    sec = next((s for s in sections if s["name"] == name), None)
    if sec and sec.get("manager") == person:
        db.set_section_manager(name, "")
    return jsonify({"ok": True})


@app.route("/api/sections/<name>/manager", methods=["POST"])
def api_sections_set_manager(name):
    data = request.get_json(silent=True) or {}
    person = (data.get("person") or "").strip()
    # person == "" clears the manager
    db.set_section_manager(name, person)
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Zones API
# ---------------------------------------------------------------------------

@app.route("/api/zones", methods=["GET"])
def api_zones_list():
    _require_api_key()
    if not db.is_available():
        return jsonify({"ok": False, "error": "DB unavailable"}), 503
    branch = request.args.get("branch") or None
    return jsonify({"ok": True, "zones": db.get_zones(branch=branch)})


@app.route("/api/zones", methods=["POST"])
def api_zones_create():
    _require_api_key()
    if not db.is_available():
        return jsonify({"ok": False, "error": "DB unavailable"}), 503
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"ok": False, "error": "name is required"}), 400
    description = (data.get("description") or "").strip()
    branch = (data.get("branch") or "Riyadh").strip()
    try:
        zone_id = db.create_zone(name, description, branch)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    return jsonify({"ok": True, "id": zone_id})


@app.route("/api/zones/<int:zone_id>", methods=["PUT"])
def api_zones_update(zone_id):
    _require_api_key()
    if not db.is_available():
        return jsonify({"ok": False, "error": "DB unavailable"}), 503
    data = request.get_json(silent=True) or {}
    db.update_zone(zone_id, name=data.get("name"), description=data.get("description"), branch=data.get("branch"))
    return jsonify({"ok": True})


@app.route("/api/zones/<int:zone_id>", methods=["DELETE"])
def api_zones_delete(zone_id):
    _require_api_key()
    if not db.is_available():
        return jsonify({"ok": False, "error": "DB unavailable"}), 503
    db.delete_zone(zone_id)
    return jsonify({"ok": True})


@app.route("/api/zones/<int:zone_id>/cameras", methods=["GET"])
def api_zones_get_cameras(zone_id):
    _require_api_key()
    if not db.is_available():
        return jsonify({"ok": False, "error": "DB unavailable"}), 503
    return jsonify({"ok": True, "cameras": db.get_zone_cameras(zone_id)})


@app.route("/api/zones/<int:zone_id>/cameras", methods=["POST"])
def api_zones_set_cameras(zone_id):
    _require_api_key()
    if not db.is_available():
        return jsonify({"ok": False, "error": "DB unavailable"}), 503
    data = request.get_json(silent=True) or {}
    location_ids = data.get("location_ids", [])
    db.set_zone_cameras(zone_id, location_ids)
    return jsonify({"ok": True})


@app.route("/api/zones/<int:zone_id>/members", methods=["GET"])
def api_zones_get_members(zone_id):
    _require_api_key()
    if not db.is_available():
        return jsonify({"ok": False, "error": "DB unavailable"}), 503
    return jsonify({"ok": True, "members": db.get_zone_members(zone_id)})


@app.route("/api/zones/<int:zone_id>/assign", methods=["POST"])
def api_zones_assign(zone_id):
    _require_api_key()
    if not db.is_available():
        return jsonify({"ok": False, "error": "DB unavailable"}), 503
    data = request.get_json(silent=True) or {}
    person_name = (data.get("person_name") or "").strip()
    if not person_name:
        return jsonify({"ok": False, "error": "person_name is required"}), 400
    db.set_person_home_zone(person_name, zone_id)
    return jsonify({"ok": True})


@app.route("/api/zones/<int:zone_id>/unassign", methods=["POST"])
def api_zones_unassign(zone_id):
    _require_api_key()
    if not db.is_available():
        return jsonify({"ok": False, "error": "DB unavailable"}), 503
    data = request.get_json(silent=True) or {}
    person_name = (data.get("person_name") or "").strip()
    if not person_name:
        return jsonify({"ok": False, "error": "person_name is required"}), 400
    db.set_person_home_zone(person_name, None)
    return jsonify({"ok": True})


@app.route("/api/zones/status", methods=["GET"])
def api_zones_status():
    _require_api_key()
    if not db.is_available():
        return jsonify({"ok": False, "error": "DB unavailable"}), 503
    cfg = _load_reports_config()
    threshold = int(cfg.get("zone_away_threshold_minutes", 30))
    rows = db.get_zone_status_snapshot(away_threshold_minutes=threshold)
    branch = request.args.get("branch") or None
    if branch:
        rows = [r for r in rows if r.get("branch") == branch]
    return jsonify({"ok": True, "status": rows})


@app.route("/api/zones/report", methods=["GET"])
def api_zones_report():
    _require_api_key()
    if not db.is_available():
        return jsonify({"ok": False, "error": "DB unavailable"}), 503
    date_from = request.args.get("date_from", "")
    date_to = request.args.get("date_to", "")
    branch = request.args.get("branch") or None
    if not date_from or not date_to:
        return jsonify({"ok": False, "error": "date_from and date_to are required"}), 400
    rows = db.get_zone_compliance_report(date_from, date_to, branch=branch)
    return jsonify({"ok": True, "date_from": date_from, "date_to": date_to, "rows": rows})


@app.route("/api/person/<name>/meta", methods=["GET"])
def api_person_meta_get(name):
    person = safe_person_name(name)
    meta = db.get_person_meta(person) or {"name": person, "section": "", "branch": "Riyadh"}
    return jsonify({"ok": True, **meta})


@app.route("/api/person/<name>/meta", methods=["POST"])
def api_person_meta_set(name):
    person = safe_person_name(name)
    if not person:
        return jsonify({"ok": False, "error": "Invalid person name"}), 400
    data = request.get_json(silent=True) or {}
    section      = data.get("section")
    branch       = data.get("branch")
    email        = data.get("email")
    arabic_name  = data.get("arabic_name")
    shift        = data.get("shift")
    if section is None and branch is None and email is None and arabic_name is None and shift is None:
        return jsonify({"ok": False, "error": "Provide section, branch, email, arabic_name, and/or shift"}), 400
    db.upsert_person_meta(person, section=section, branch=branch, email=email, arabic_name=arabic_name, shift=shift)
    return jsonify({"ok": True, "name": person, "section": section, "branch": branch, "email": email, "arabic_name": arabic_name, "shift": shift})

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

@app.route("/api/bulk_upload_faces", methods=["POST"])
def api_bulk_upload_faces():
    """
    Bulk upload face images.

    form-data:
      - bulk_mode: "single_person" | "name_from_file"
      - existing_name / new_name / mode: (single_person only) same as upload_face
      - files[]: one or more image files
    """
    files = request.files.getlist("files[]")
    if not files:
        return jsonify({"ok": False, "error": "no files provided"}), 400

    bulk_mode = (request.form.get("bulk_mode") or "single_person").strip().lower()
    results = []
    errors = []

    if bulk_mode == "single_person":
        mode = (request.form.get("mode") or "existing").strip().lower()
        existing_name = (request.form.get("existing_name") or "").strip()
        new_name = (request.form.get("new_name") or "").strip()
        person = safe_person_name(existing_name if mode == "existing" else new_name)
        if not person:
            return jsonify({"ok": False, "error": "person name is required"}), 400
        if mode == "new":
            person_dir_check = os.path.join(FACES_DIR, person)
            if os.path.isdir(person_dir_check):
                return jsonify({"ok": False, "error": f"Person '{person}' already exists. Use 'Existing' to add more images."}), 409
        person_dir = os.path.join(FACES_DIR, person)
        os.makedirs(person_dir, exist_ok=True)
        for f in files:
            original = secure_filename(f.filename or "")
            ext = os.path.splitext(original)[1].lower()
            if ext not in ALLOWED_EXTS:
                ext = ".jpg"
            filename = next_image_filename(person_dir, ext)
            path = os.path.join(person_dir, filename)
            f.save(path)
            results.append({"person": person, "saved": f"/faces/{person}/{filename}"})

    elif bulk_mode == "name_from_file":
        for f in files:
            original = secure_filename(f.filename or "")
            stem, ext = os.path.splitext(original)
            ext = ext.lower()
            if ext not in ALLOWED_EXTS:
                ext = ".jpg"
            person = safe_person_name(stem)
            if not person:
                errors.append({"file": original, "error": "could not derive person name from filename"})
                continue
            person_dir = os.path.join(FACES_DIR, person)
            os.makedirs(person_dir, exist_ok=True)
            filename = next_image_filename(person_dir, ext)
            path = os.path.join(person_dir, filename)
            f.save(path)
            results.append({"person": person, "saved": f"/faces/{person}/{filename}"})
    else:
        return jsonify({"ok": False, "error": "invalid bulk_mode"}), 400

    if results:
        _reload_faces_async()
    return jsonify({"ok": True, "uploaded": len(results), "results": results, "errors": errors})

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
        db.rename_person_meta(old_name, new_name)
        _reload_faces_async()
        return jsonify({"ok": True, "person": new_name, "renamed_visits": rows,
                        "merged": True, "images_moved": moved})

    # Simple rename
    os.rename(old_dir, new_dir)
    rows = db.rename_person(old_name, new_name)
    db.rename_person_meta(old_name, new_name)
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

    # Delete visits and metadata from DB
    deleted_visits = db.delete_person_visits(person)
    db.delete_person_meta(person)

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
    if len(sources) < 3:
        log.warning("build_analysis_pool_source: only %d camera(s) configured — expected 17", len(sources))
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

@app.route("/api/camera/statuses", methods=["GET"])
def api_camera_statuses():
    """Return live/dead status for all cameras in the active grid."""
    try:
        statuses = engine.get_camera_statuses()
    except Exception:
        statuses = {}
    return jsonify(statuses)

@app.route("/api/camera/reconnect", methods=["POST"])
def api_camera_reconnect():
    """Bypass the 2-minute reconnect backoff and immediately retry one camera.

    Body: {"source": "<rtsp url or camera source string>"}
    """
    data = request.get_json(silent=True) or {}
    source = data.get("source", "").strip()
    if not source:
        return jsonify({"ok": False, "error": "source required"}), 400
    found = engine.force_reconnect_camera(source)
    if not found:
        return jsonify({"ok": False, "error": "camera not found in active grid"}), 404
    return jsonify({"ok": True})

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
    branch = str(payload.get("branch", "Riyadh")).strip() or "Riyadh"
    with _ip_cameras_lock:
        state = _load_ip_cameras()
        group = {"id": _new_id(), "name": name, "base_url": base_url, "branch": branch, "cameras": []}
        state["groups"].append(group)
        _save_ip_cameras(state)
    return jsonify({"ok": True, "group": group})


@app.route("/api/ip_cameras/groups/<group_id>", methods=["PUT"])
def api_ip_cameras_group_update(group_id):
    payload = request.get_json(silent=True) or {}
    new_name = payload.get("name")
    new_base = payload.get("base_url")
    new_branch = payload.get("branch")
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
        if new_branch is not None:
            b = str(new_branch).strip()
            if b:
                g["branch"] = b
        _save_ip_cameras(state)
    return jsonify({"ok": True, "group": {
        "id": g["id"], "name": g["name"], "base_url": g["base_url"],
        "branch": g.get("branch") or "Riyadh",
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
    """Convert a value to datetime — handles both datetime objects and ISO strings.

    Timestamps are stored as bare local-time strings (no tzinfo suffix).
    fromisoformat() returns a naive datetime; we attach the local timezone so
    that .astimezone() calls elsewhere produce the correct local time instead
    of shifting by the UTC offset.
    """
    if val is None:
        return None
    if isinstance(val, datetime):
        if val.tzinfo is None:
            return val.astimezone()  # attach local tz
        return val
    if isinstance(val, str):
        dt = datetime.fromisoformat(val)
        if dt.tzinfo is None:
            dt = dt.astimezone()  # treat bare string as local time
        return dt
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

    branch = request.args.get("branch") or None
    visits = db.get_daily_summary(day, branch=branch)
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

    branch = request.args.get("branch") or None
    visits = db.get_person_visits(name, date_from=df, date_to=dt, branch=branch)
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

    branch = request.args.get("branch") or None
    visits = db.get_location_visits(location_id, date_from=df, date_to=dt, branch=branch)
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
    branch = request.args.get("branch") or None
    return jsonify({"ok": True, "persons": db.get_known_persons(branch=branch)})


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


# ── Camera Tracker ───────────────────────────────────────────────────────────

@app.route("/tracker")
def tracker_page():
    return render_template("tracker.html", api_key=API_KEY)


@app.route("/api/tracker/config", methods=["GET"])
def api_tracker_config_get():
    cfg = _load_tracker_config()
    nm  = engine.source_name_map if engine else {}
    return jsonify({
        "cameras":        [{"source": s, "name": nm.get(s, s)} for s in cfg["cameras"]],
        "line_y_ratio":   cfg["line_y_ratio"],
        "line_y_ratios":  cfg.get("line_y_ratios", {}),
        "cam_transforms": cfg.get("cam_transforms", {}),
        "tracker_rois":   cfg.get("tracker_rois", {}),
    })


@app.route("/api/tracker/config", methods=["POST"])
def api_tracker_config_set():
    payload = request.get_json(silent=True) or {}
    cameras = [str(s).strip() for s in (payload.get("cameras") or []) if str(s).strip()][:4]
    line_y  = max(0.05, min(0.95, float(payload.get("line_y_ratio", 0.5))))
    # Per-camera line positions: {source: ratio}
    raw_ratios = payload.get("line_y_ratios") or {}
    line_y_ratios = {str(k): max(0.05, min(0.95, float(v)))
                     for k, v in raw_ratios.items() if str(k) and v is not None}
    known   = set(_all_configured_sources())
    invalid = [c for c in cameras if c not in known]
    if invalid:
        return jsonify({"ok": False, "error": f"Unknown camera sources: {invalid}"}), 400
    raw_transforms = payload.get("cam_transforms") or {}
    cam_transforms = {}
    for src, t in raw_transforms.items():
        if not isinstance(t, dict):
            continue
        cam_transforms[str(src)] = {
            "zoom":   max(0.5, min(8.0, float(t.get("zoom", 1.0)))),
            "panX":   float(t.get("panX", 0.0)),
            "panY":   float(t.get("panY", 0.0)),
            "rotate": int(t.get("rotate", 0)) % 360,
        }
    raw_rois = payload.get("tracker_rois") or {}
    tracker_rois = {}
    for src, r in raw_rois.items():
        if isinstance(r, (list, tuple)) and len(r) == 4:
            tracker_rois[str(src)] = [max(0.0, min(1.0, float(v))) for v in r]
    cfg = {"cameras": cameras, "line_y_ratio": line_y, "line_y_ratios": line_y_ratios,
           "cam_transforms": cam_transforms, "tracker_rois": tracker_rois}
    try:
        _save_tracker_config(cfg)
    except Exception as _e:
        return jsonify({"ok": False, "error": str(_e)}), 500
    try:
        if engine:
            engine.set_tracker_cameras(cameras)
            engine.set_config({"tracker_line_y_ratios": line_y_ratios,
                               "tracker_rois": tracker_rois})
    except Exception as _e:
        log.warning("set_tracker_cameras: %s", _e)
    return jsonify({"ok": True, "config": cfg})


@app.route("/api/tracker/events")
def api_tracker_events():
    if not db.is_available():
        return jsonify({"ok": False, "error": "database not available"}), 503
    limit  = min(int(request.args.get("limit", 200)), 500)
    offset = int(request.args.get("offset", 0))
    rows   = db.get_tracker_events(limit=limit, offset=offset)
    for r in rows:
        r["snapshot_url"] = f"/{r['snapshot_path']}" if r.get("snapshot_path") else None
    return jsonify({"ok": True, "events": rows})


@app.route("/api/tracker/ping", methods=["POST"])
def api_tracker_ping():
    """Keep-alive from the tracker page. Marks tracker as active so the main
    composite stream yields bandwidth to per-camera streams."""
    global _tracker_active_t
    _tracker_active_t = time.monotonic()
    return jsonify({"ok": True})


@app.route("/api/tracker/stream")
def api_tracker_stream():
    def gen():
        try:
            rows = db.get_tracker_events(limit=20)
            for r in rows:
                r["snapshot_url"] = f"/{r['snapshot_path']}" if r.get("snapshot_path") else None
            yield f"event: snapshot\ndata: {json.dumps(rows)}\n\n"
        except Exception:
            pass
        while True:
            try:
                msg = tracker_events_q.get(timeout=20)
            except Empty:
                yield ": ping\n\n"
                continue
            yield f"event: {msg['event']}\ndata: {json.dumps(msg['data'])}\n\n"

    return Response(
        stream_with_context(gen()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
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
    branch = request.args.get("branch") or None

    ds = day_start.isoformat() if db._backend == "sqlite" else day_start
    de = day_end.isoformat()   if db._backend == "sqlite" else day_end

    branch_clause = f"AND branch = {ph}" if branch else ""

    # Peak hour — hour bucket (local time) with the most visit records
    if db._backend == "sqlite":
        hour_sql = f"""
            SELECT strftime('%H', datetime(first_seen, 'localtime')) as hr,
                   COUNT(DISTINCT person_name) as cnt
            FROM visits
            WHERE first_seen >= {ph} AND first_seen < {ph}
            {branch_clause}
            GROUP BY hr ORDER BY cnt DESC LIMIT 1
        """
    else:
        hour_sql = f"""
            SELECT date_part('hour', first_seen AT TIME ZONE 'localtime') as hr,
                   COUNT(DISTINCT person_name) as cnt
            FROM visits
            WHERE first_seen >= {ph} AND first_seen < {ph}
            {branch_clause}
            GROUP BY hr ORDER BY cnt DESC LIMIT 1
        """

    # People present today — distinct known persons seen today
    present_sql = f"""
        SELECT COUNT(DISTINCT person_name) as cnt FROM visits
        WHERE first_seen >= {ph} AND first_seen < {ph}
          AND person_name NOT LIKE 'unknown_%'
          {branch_clause}
    """

    base_params = [ds, de]
    if branch:
        base_params.append(branch)
    params = tuple(base_params)
    with db._cursor() as cur:
        cur.execute(hour_sql, params)
        hour_row = db._row_to_dict(cur.fetchone())

        cur.execute(present_sql, params)
        present_row = db._row_to_dict(cur.fetchone())

    # Count unknowns from faces/ folder
    known_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faces")
    import re as _re
    unknowns_count = 0
    if os.path.isdir(known_dir):
        for d in os.listdir(known_dir):
            if os.path.isdir(os.path.join(known_dir, d)) and _re.match(r'^unknown_\d+$', d):
                unknowns_count += 1

    present_today = int(present_row["cnt"]) if present_row else 0

    # Absent = people assigned to this branch in the people table minus present today.
    # Without branch filter, fall back to total enrolled known folders.
    if branch:
        enrolled_known = len(db.get_branch_members(branch))
    else:
        enrolled_known = 0
        if os.path.isdir(known_dir):
            for d in os.listdir(known_dir):
                if os.path.isdir(os.path.join(known_dir, d)) and not _re.match(r'^unknown_\d+$', d):
                    enrolled_known += 1

    absent_today = max(0, enrolled_known - present_today)

    peak_hour = None
    peak_hour_count = 0
    if hour_row and hour_row.get("hr") is not None:
        h = int(hour_row["hr"])
        peak_hour = f"{h:02d}:00 – {(h + 1) % 24:02d}:00"
        peak_hour_count = int(hour_row["cnt"])

    return jsonify({
        "ok": True,
        "date": date_str,
        "peak_hour":      peak_hour,
        "peak_hour_count": peak_hour_count,
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
    branch = request.args.get("branch") or None
    ds = day_start.isoformat() if db._backend == "sqlite" else day_start
    de = day_end.isoformat()   if db._backend == "sqlite" else day_end

    branch_clause = f"AND branch = {ph}" if branch else ""
    base_params = [ds, de]
    if branch:
        base_params.append(branch)

    present_sql = f"""
        SELECT DISTINCT person_name FROM visits
        WHERE first_seen >= {ph} AND first_seen < {ph}
          AND person_name NOT LIKE 'unknown_%'
          {branch_clause}
        ORDER BY person_name
    """
    with db._cursor() as cur:
        cur.execute(present_sql, tuple(base_params))
        present = [r["person_name"] for r in db._rows_to_dicts(cur.fetchall())]

    # Absent = persons assigned to this branch in the people table who didn't appear today.
    # Falls back to all enrolled known folders when no branch filter.
    import re as _re
    present_set = set(present)
    if branch:
        branch_members = db.get_branch_members(branch)
        absent = [n for n in branch_members if n not in present_set]
    else:
        known_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faces")
        absent = []
        if os.path.isdir(known_dir):
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
          morning = work_start - 1h → work_end        (today)
          night   = night_work_start - 1h (today) → night_work_end (tomorrow)
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
    branch = request.args.get("branch") or None
    ph = "?" if db._backend == "sqlite" else "%s"

    _rcfg = _load_reports_config()
    _morning_h, _morning_m = map(int, _rcfg.get("work_start", "09:00").split(":"))
    _morning_end_h, _morning_end_m = map(int, _rcfg.get("work_end", "17:00").split(":"))
    _night_h,   _night_m   = map(int, _rcfg.get("night_work_start", "21:00").split(":"))
    _night_end_h, _night_end_m = map(int, _rcfg.get("night_work_end", "05:00").split(":"))

    if shift == "morning":
        # Include arrivals starting 1 hour before shift start
        start = (day + timedelta(hours=_morning_h, minutes=_morning_m) - timedelta(hours=1)).astimezone(timezone.utc)
        end   = (day + timedelta(hours=_morning_end_h, minutes=_morning_end_m)).astimezone(timezone.utc)
    elif shift == "night":
        # Night shift starts this evening and ends tomorrow morning:
        #   night_work_start - 1h (today)  →  night_work_end (tomorrow)
        # e.g. today 20:00 → tomorrow 05:00.
        start = (day + timedelta(hours=_night_h, minutes=_night_m) - timedelta(hours=1)).astimezone(timezone.utc)
        end   = (day + timedelta(days=1, hours=_night_end_h, minutes=_night_end_m)).astimezone(timezone.utc)
    else:
        start = day.astimezone(timezone.utc)
        end   = (day + timedelta(days=1)).astimezone(timezone.utc)

    s_val = start.isoformat() if db._backend == "sqlite" else start
    e_val = end.isoformat()   if db._backend == "sqlite" else end
    params = [s_val, e_val]
    if branch:
        params.append(branch)
    branch_clause = f"AND branch = {ph}" if branch else ""

    # For night shift: exclude anyone who already arrived during morning shift
    # so each person appears in at most one shift row.
    morning_exclusion = ""
    morning_params = []
    if shift == "night":
        morning_start = (day + timedelta(hours=_morning_h, minutes=_morning_m)).astimezone(timezone.utc)
        morning_end   = (day + timedelta(hours=_morning_end_h, minutes=_morning_end_m)).astimezone(timezone.utc)
        ms = morning_start.isoformat() if db._backend == "sqlite" else morning_start
        me = morning_end.isoformat()   if db._backend == "sqlite" else morning_end
        morning_exclusion = f"""
          AND person_name NOT IN (
            SELECT DISTINCT person_name FROM visits
            WHERE first_seen >= {ph} AND first_seen < {ph}
          )"""
        morning_params = [ms, me]

    sql = f"""
        SELECT person_name, MIN(first_seen) as earliest
        FROM visits
        WHERE first_seen >= {ph} AND first_seen < {ph}
          AND person_name NOT LIKE 'unknown_%'
          {branch_clause}
          {morning_exclusion}
        GROUP BY person_name
        ORDER BY earliest {order}
        LIMIT 10
    """
    with db._cursor() as cur:
        cur.execute(sql, tuple(params) + tuple(morning_params))
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
    branch = request.args.get("branch") or None
    branch_clause = f"AND branch = {ph}" if branch else ""
    params = [start_val]
    if branch:
        params.append(branch)

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
          {branch_clause}
        GROUP BY person_name
        ORDER BY total_secs DESC
        LIMIT 10
    """
    with db._cursor() as cur:
        cur.execute(sql, tuple(params))
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
    branch = request.args.get("branch") or None
    branch_clause = f"AND branch = {ph}" if branch else ""
    params = [from_date, to_date]
    if branch:
        params.append(branch)

    if db._backend == "sqlite":
        date_expr = "DATE(first_seen)"
    else:
        date_expr = "first_seen::date"

    sql = f"""
        SELECT {date_expr} as day, COUNT(DISTINCT person_name) as count
        FROM visits
        WHERE {date_expr} >= {ph} AND {date_expr} <= {ph}
          AND person_name NOT LIKE 'unknown_%'
          {branch_clause}
        GROUP BY day
        ORDER BY day ASC
    """
    with db._cursor() as cur:
        cur.execute(sql, tuple(params))
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
    branch = request.args.get("branch") or None
    branch_clause = f"AND branch = {ph}" if branch else ""
    params = [from_date, to_date]
    if branch:
        params.append(branch)

    if db._backend == "sqlite":
        date_expr = "DATE(first_seen)"
    else:
        date_expr = "first_seen::date"

    sql = f"""
        SELECT {date_expr} as day, person_name
        FROM visits
        WHERE {date_expr} >= {ph} AND {date_expr} <= {ph}
          AND person_name NOT LIKE 'unknown_%'
          {branch_clause}
        GROUP BY day, person_name
        ORDER BY LOWER(person_name) ASC, day ASC
    """
    with db._cursor() as cur:
        cur.execute(sql, tuple(params))
        rows = db._rows_to_dicts(cur.fetchall())

    dates_set = sorted({r["day"] for r in rows})
    persons_set = sorted({r["person_name"] for r in rows}, key=str.lower)
    present = {p: {} for p in persons_set}
    for r in rows:
        present[r["person_name"]][r["day"]] = True

    return jsonify({"ok": True, "dates": dates_set, "persons": persons_set, "present": present})


# ---------------------------------------------------------------------------
# Gate Reports
# ---------------------------------------------------------------------------

_REPORTS_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports_config.json")


def _load_reports_config():
    if os.path.exists(_REPORTS_CONFIG_PATH):
        try:
            with open(_REPORTS_CONFIG_PATH) as f:
                cfg = json.load(f)
            # Migrate legacy single gate_camera → arrival_camera
            if "gate_camera" in cfg and "arrival_camera" not in cfg:
                cfg["arrival_camera"] = cfg.pop("gate_camera")
            return cfg
        except Exception:
            pass
    return {
        "arrival_camera": "",
        "exit_camera": "",
        "manager_email": "",
        "work_start": "08:00",
        "late_threshold_minutes": 15,
        "daily_send_time": "17:00",
        "night_shift_enabled": False,
        "night_work_start": "21:00",
        "night_late_threshold_minutes": 15,
    }


def _save_reports_config(cfg):
    with open(_REPORTS_CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def _generate_gate_report(arrival_camera, exit_camera, date_from, date_to,
                          late_threshold_minutes=15, work_start="08:00",
                          night_shift_enabled=False, night_work_start="21:00",
                          night_late_threshold_minutes=15):
    """Build per-person daily report from gate_events + arrival visits.

    gate_events  → exits, returns, durations (written live by _handle_gate_event)
    visits table → arrival time (first seen on arrival_camera each day)
    Each person's shift (morning/night) is read from their profile; the matching
    work_start and late threshold are used for that person's row.
    """
    from collections import defaultdict

    date_from_str = date_from.astimezone().strftime("%Y-%m-%d")
    date_to_str = (date_to - timedelta(seconds=1)).astimezone().strftime("%Y-%m-%d")

    # Raw gate events for the date range
    raw_events = db.get_gate_events_range(date_from_str, date_to_str)

    # Arrival times: earliest visit on ANY camera per person per day
    arrival_map = defaultdict(dict)  # person → {date_str → HH:MM}
    ph = "?" if db._backend == "sqlite" else "%s"
    if db._backend == "sqlite":
        arr_params = [date_from.isoformat(), date_to.isoformat()]
    else:
        arr_params = [date_from, date_to]
    arr_sql = f"""
        SELECT person_name, MIN(first_seen) as first_seen
        FROM visits
        WHERE first_seen >= {ph} AND first_seen < {ph}
          AND person_name NOT LIKE 'unknown_%'
        GROUP BY person_name, date(first_seen)
    """
    with db._cursor() as cur:
        cur.execute(arr_sql, arr_params)
        for row in db._rows_to_dicts(cur.fetchall()):
            fs = _to_dt(row["first_seen"])
            if fs:
                day = fs.astimezone().strftime("%Y-%m-%d")
                arrival_map[row["person_name"]][day] = fs.astimezone().strftime("%H:%M")

    all_meta = db.get_all_people_meta()
    meta_map = {m["name"]: m.get("arabic_name", "") for m in all_meta}
    shift_map = {m["name"]: (m.get("shift") or "morning") for m in all_meta}
    all_people = {m["name"] for m in all_meta}
    work_h, work_m = map(int, work_start.split(":"))
    night_h, night_m = map(int, night_work_start.split(":"))

    # Group events by person+date
    by_person_day = defaultdict(lambda: defaultdict(list))
    for ev in raw_events:
        by_person_day[ev["person_name"]][ev["event_date"]].append(ev)

    # Include people who only have an arrival (no exits yet)
    for person, days in arrival_map.items():
        for day in days:
            by_person_day[person][day]  # touch to create entry

    # Add absent employees for every date in the report range
    report_dates = set()
    d = date_from.astimezone()
    end = date_to.astimezone()
    while d < end:
        report_dates.add(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)

    for person in all_people:
        for day_str in report_dates:
            by_person_day[person][day_str]  # touch — creates empty list if not present

    records = []
    for person in sorted(by_person_day, key=lambda n: n.lower()):
        for day_str in sorted(by_person_day[person]):
            events = sorted(by_person_day[person][day_str], key=lambda e: e["exit_time"] or "")

            arrival_str = arrival_map.get(person, {}).get(day_str, "")
            person_shift = shift_map.get(person, "morning")
            if night_shift_enabled and person_shift == "night":
                p_work_h, p_work_m = night_h, night_m
                p_late_threshold = night_late_threshold_minutes
            else:
                p_work_h, p_work_m = work_h, work_m
                p_late_threshold = late_threshold_minutes

            # Determine arrived_late
            if arrival_str:
                arr_h, arr_m = map(int, arrival_str.split(":"))
                arrived_late = (arr_h, arr_m) > (p_work_h, p_work_m)
                late_arrival_min = round(((arr_h * 60 + arr_m) - (p_work_h * 60 + p_work_m)), 1) if arrived_late else 0.0
            else:
                arrived_late = False
                late_arrival_min = 0.0

            exit_events = []
            last_exit_str = ""
            for ev in events:
                exit_t = ev.get("exit_time", "")
                entry_t = ev.get("entry_time")
                dur = ev.get("duration_minutes")
                still_out = entry_t is None

                # Format timestamps to HH:MM
                def _fmt_ts(ts):
                    if not ts:
                        return "—"
                    try:
                        dt = _to_dt(str(ts))
                        if dt:
                            return dt.astimezone().strftime("%H:%M")
                    except Exception:
                        pass
                    return str(ts)[:5] if ts else "—"

                out_at = _fmt_ts(exit_t)
                back_at = _fmt_ts(entry_t) if not still_out else "—"
                if out_at != "—":
                    last_exit_str = out_at

                exit_events.append({
                    "out_at": out_at,
                    "back_at": back_at,
                    "duration_minutes": round(dur, 1) if dur is not None else None,
                    "late": (dur or 0) > p_late_threshold,
                    "still_out": still_out,
                })

            total_outside_min = round(sum(
                e["duration_minutes"] for e in exit_events
                if not e["still_out"] and e["duration_minutes"] is not None
            ), 1)

            absent = not arrival_str and not exit_events
            records.append({
                "person": person,
                "arabic_name": meta_map.get(person, ""),
                "date": day_str,
                "shift": person_shift,
                "absent": absent,
                "arrival": arrival_str or "—",
                "arrived_late": arrived_late,
                "late_arrival_minutes": late_arrival_min,
                "last_exit": last_exit_str or "—",
                "exits": exit_events,
                "late_exits_count": sum(1 for e in exit_events if e["late"] and not e["still_out"]),
                "total_outside_minutes": total_outside_min,
            })

    return records


def _build_report_excel(records, report_date_label, camera_name, work_start, late_threshold):
    """Build an Arabic Excel (.xlsx) report and return raw bytes."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side, numbers
    from openpyxl.utils import get_column_letter
    from io import BytesIO

    wb = Workbook()
    ws = wb.active
    ws.title = "تقرير البوابة"
    ws.sheet_view.rightToLeft = True

    # ── Palette ──────────────────────────────────────────────
    HDR_FILL   = PatternFill("solid", fgColor="1E3A5F")
    META_FILL  = PatternFill("solid", fgColor="2D6A4F")
    RED_FILL    = PatternFill("solid", fgColor="FEE2E2")
    AMBER_FILL  = PatternFill("solid", fgColor="FEF3C7")
    OK_FILL     = PatternFill("solid", fgColor="D1FAE5")
    ALT_FILL    = PatternFill("solid", fgColor="F8FAFC")
    EXIT_FILL   = PatternFill("solid", fgColor="EFF6FF")
    ABSENT_FILL = PatternFill("solid", fgColor="F1F5F9")
    thin = Side(style="thin", color="CBD5E1")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    def hdr_cell(ws, row, col, value, fill=HDR_FILL, font_size=11):
        c = ws.cell(row=row, column=col, value=value)
        c.font = Font(bold=True, color="FFFFFF", size=font_size, name="Cairo")
        c.fill = fill
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = border
        return c

    def data_cell(ws, row, col, value, fill=None, bold=False, color="111111", align="center"):
        c = ws.cell(row=row, column=col, value=value)
        c.font = Font(bold=bold, color=color, size=10, name="Cairo")
        if fill:
            c.fill = fill
        c.alignment = Alignment(horizontal=align, vertical="center", wrap_text=True)
        c.border = border
        return c

    # ── Title row ────────────────────────────────────────────
    ws.merge_cells("A1:K1")
    title = ws["A1"]
    title.value = f"تقرير البوابة اليومي — {report_date_label}"
    title.font = Font(bold=True, size=14, color="FFFFFF", name="Cairo")
    title.fill = PatternFill("solid", fgColor="0F172A")
    title.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 32

    # ── Meta row ─────────────────────────────────────────────
    ws.merge_cells("A2:K2")
    meta = ws["A2"]
    meta.value = f"الكاميرا: {camera_name}   |   بداية الدوام: {work_start}   |   حد التأخير: {late_threshold} دقيقة"
    meta.font = Font(bold=False, size=10, color="FFFFFF", name="Cairo")
    meta.fill = META_FILL
    meta.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[2].height = 22

    # ── Column headers ────────────────────────────────────────
    # Col: 1=#  2=الاسم  3=التاريخ  4=وقت الحضور  5=آخر مغادرة
    #      6=عدد المغادرات  7=خروج  8=دخول  9=مدة الغياب  10=تأخر؟
    #      11=إجمالي وقت الغياب  (status moved to last: but merged with total)
    #      Actually: 11=إجمالي وقت الغياب  12=الحالة  → 12 cols still
    headers = [
        "#", "الاسم", "التاريخ",
        "وقت الحضور", "آخر مغادرة", "عدد المغادرات",
        "خروج", "دخول", "مدة الغياب (د)", "تأخر؟",
        "إجمالي وقت الغياب (د)", "الحالة",
    ]
    for col, h in enumerate(headers, 1):
        hdr_cell(ws, 3, col, h)
    ws.row_dimensions[3].height = 28

    # ── Data rows ─────────────────────────────────────────────
    row_num = 4
    for idx, r in enumerate(records, 1):
        has_issue = r["arrived_late"] or r["late_exits_count"] > 0
        row_fill = RED_FILL if has_issue else (ALT_FILL if idx % 2 == 0 else None)

        if r["absent"]:
            arrival_status = "غائب"
            status_fill = ABSENT_FILL
            status_color = "6B7280"
        elif r["arrived_late"]:
            arrival_status = "متأخر"
            status_fill = RED_FILL
            status_color = "991B1B"
        else:
            arrival_status = "في الوقت"
            status_fill = OK_FILL
            status_color = "166534"

        exits = r.get("exits", [])
        n_exits = len(exits)
        last_exit = (exits[-1]["back_at"] or exits[-1]["out_at"]) if exits else "—"

        if n_exits == 0:
            data_cell(ws, row_num, 1,  idx,                           row_fill)
            data_cell(ws, row_num, 2,  r["person"].replace("_", " "), row_fill)
            data_cell(ws, row_num, 3,  r["date"],                     row_fill)
            data_cell(ws, row_num, 4,  r["arrival"],                  row_fill)
            data_cell(ws, row_num, 5,  "—",                           row_fill)
            data_cell(ws, row_num, 6,  0,                             row_fill)
            data_cell(ws, row_num, 7,  "—",                           row_fill)
            data_cell(ws, row_num, 8,  "—",                           row_fill)
            data_cell(ws, row_num, 9,  "—",                           row_fill)
            data_cell(ws, row_num, 10, "—",                           row_fill)
            data_cell(ws, row_num, 11, 0,                             row_fill)
            data_cell(ws, row_num, 12, arrival_status, status_fill, color=status_color, bold=True)
            row_num += 1
        else:
            for ei, ex in enumerate(exits):
                ex_fill = EXIT_FILL if ei % 2 == 0 else ALT_FILL
                late_fill = RED_FILL if ex["late"] else OK_FILL
                late_color = "991B1B" if ex["late"] else "166534"
                if ei == 0:
                    data_cell(ws, row_num, 1, idx,                           row_fill)
                    data_cell(ws, row_num, 2, r["person"].replace("_", " "), row_fill)
                    data_cell(ws, row_num, 3, r["date"],                      row_fill)
                    data_cell(ws, row_num, 4, r["arrival"],                   row_fill)
                    data_cell(ws, row_num, 5, last_exit,                      row_fill)
                    data_cell(ws, row_num, 6, n_exits,                        row_fill)
                else:
                    for col in range(1, 7):
                        data_cell(ws, row_num, col, "", ex_fill)
                data_cell(ws, row_num, 7,  ex["out_at"],           ex_fill)
                data_cell(ws, row_num, 8,  ex["back_at"],          ex_fill)
                data_cell(ws, row_num, 9,  ex["duration_minutes"], ex_fill)
                data_cell(ws, row_num, 10, "نعم" if ex["late"] else "لا", late_fill, color=late_color, bold=ex["late"])
                data_cell(ws, row_num, 11, r["total_outside_minutes"] if ei == 0 else "", ex_fill)
                data_cell(ws, row_num, 12, arrival_status if ei == 0 else "", status_fill if ei == 0 else ex_fill, color=status_color if ei == 0 else "111111", bold=ei == 0)
                row_num += 1

    # ── Summary footer ────────────────────────────────────────
    row_num += 1
    ws.merge_cells(f"A{row_num}:F{row_num}")
    total_late = sum(1 for r in records if r["arrived_late"] or r["late_exits_count"] > 0)
    summary = ws[f"A{row_num}"]
    summary.value = f"الإجمالي: {len(records)} موظف   |   مخالفات: {total_late}   |   إجمالي وقت الغياب: {round(sum(r['total_outside_minutes'] for r in records), 1)} دقيقة"
    summary.font = Font(bold=True, size=10, color="FFFFFF", name="Cairo")
    summary.fill = PatternFill("solid", fgColor="0F172A")
    summary.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[row_num].height = 22

    # ── Column widths ─────────────────────────────────────────
    col_widths = [5, 22, 12, 10, 10, 10, 9, 9, 14, 8, 20, 14]
    for i, w in enumerate(col_widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w

    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()


@app.route("/api/reports/config", methods=["GET"])
def api_reports_config_get():
    return jsonify({"ok": True, "config": _load_reports_config()})


@app.route("/api/reports/config", methods=["POST"])
def api_reports_config_post():
    data = request.get_json(silent=True) or {}
    cfg = _load_reports_config()
    for key in ("arrival_camera", "exit_camera", "manager_email", "cc_emails", "work_start", "late_threshold_minutes", "daily_send_time", "night_shift_enabled", "night_work_start", "night_late_threshold_minutes"):
        if key in data:
            cfg[key] = data[key]
    # Drop legacy key if still present
    cfg.pop("gate_camera", None)
    _save_reports_config(cfg)
    return jsonify({"ok": True, "config": cfg})


@app.route("/api/advanced/config", methods=["GET"])
def api_advanced_config_get():
    return jsonify({"ok": True, "config": _load_reports_config()})


@app.route("/api/advanced/config", methods=["POST"])
def api_advanced_config_post():
    data = request.get_json(silent=True) or {}
    cfg = _load_reports_config()
    for key in ("work_start", "work_end", "late_threshold_minutes",
                "night_shift_enabled", "night_work_start",
                "night_work_end", "night_late_threshold_minutes"):
        if key in data:
            cfg[key] = data[key]
    _save_reports_config(cfg)
    return jsonify({"ok": True, "config": cfg})


@app.route("/api/reports/gate", methods=["GET"])
def api_reports_gate():
    """Generate gate attendance report using arrival + exit cameras."""
    if not db.is_available():
        return jsonify({"ok": False, "error": "database not available"}), 503

    cfg = _load_reports_config()
    arrival_cam = request.args.get("arrival_camera") or cfg.get("arrival_camera", "")
    exit_cam = request.args.get("exit_camera") or cfg.get("exit_camera", "")
    if not arrival_cam and not exit_cam:
        return jsonify({"ok": False, "error": "No arrival or exit camera configured"}), 400

    late_threshold = int(cfg.get("late_threshold_minutes", 15))
    work_start = cfg.get("work_start", "08:00")
    night_shift_enabled = bool(cfg.get("night_shift_enabled", False))
    night_work_start = cfg.get("night_work_start", "21:00")
    night_late_threshold = int(cfg.get("night_late_threshold_minutes", 15))

    days_param = request.args.get("days")
    if days_param:
        try:
            n = int(days_param)
        except ValueError:
            return jsonify({"ok": False, "error": "invalid days param"}), 400
        date_to = datetime.now(timezone.utc)
        date_from = date_to - timedelta(days=n)
    else:
        date_str = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
        try:
            local_midnight = datetime.strptime(date_str, "%Y-%m-%d").astimezone()
        except ValueError:
            return jsonify({"ok": False, "error": "invalid date format"}), 400
        date_from = local_midnight.astimezone(timezone.utc)
        date_to = date_from + timedelta(days=1)

    records = _generate_gate_report(arrival_cam, exit_cam, date_from, date_to, late_threshold, work_start,
                                    night_shift_enabled, night_work_start, night_late_threshold)
    loc = db.get_location_by_source(arrival_cam or exit_cam)
    camera_name = loc["name"] if loc else (arrival_cam or exit_cam)

    return jsonify({
        "ok": True,
        "arrival_camera": arrival_cam,
        "exit_camera": exit_cam,
        "camera_name": camera_name,
        "work_start": work_start,
        "late_threshold_minutes": late_threshold,
        "records": records,
        "summary": {
            "total": len(records),
            "absent": sum(1 for r in records if r.get("absent")),
            "late_arrival": sum(1 for r in records if r["arrived_late"]),
            "late_exits": sum(1 for r in records if r["late_exits_count"] > 0),
        },
    })


@app.route("/api/reports/export", methods=["GET"])
def api_reports_export():
    """Export gate report as Arabic Excel (.xlsx)."""
    if not db.is_available():
        return jsonify({"ok": False, "error": "database not available"}), 503

    cfg = _load_reports_config()
    arrival_cam = cfg.get("arrival_camera", "")
    exit_cam = cfg.get("exit_camera", "")
    if not arrival_cam and not exit_cam:
        return jsonify({"ok": False, "error": "No cameras configured"}), 400

    late_threshold = int(cfg.get("late_threshold_minutes", 15))
    work_start = cfg.get("work_start", "08:00")
    night_shift_enabled = bool(cfg.get("night_shift_enabled", False))
    night_work_start = cfg.get("night_work_start", "21:00")
    night_late_threshold = int(cfg.get("night_late_threshold_minutes", 15))

    days_param = request.args.get("days", "1")
    try:
        n = int(days_param)
    except ValueError:
        return jsonify({"ok": False, "error": "invalid days param"}), 400

    date_to = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=n)

    records = _generate_gate_report(arrival_cam, exit_cam, date_from, date_to, late_threshold, work_start,
                                    night_shift_enabled, night_work_start, night_late_threshold)
    loc = db.get_location_by_source(arrival_cam or exit_cam)
    camera_name = loc["name"] if loc else (arrival_cam or exit_cam)
    date_label = f"آخر {n} يوم" if n > 1 else datetime.now().strftime("%Y-%m-%d")

    xlsx_bytes = _build_report_excel(records, date_label, camera_name, work_start, late_threshold)
    response = app.make_response(xlsx_bytes)
    response.headers["Content-Type"] = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    filename = f"gate_report_{n}d_{datetime.now().strftime('%Y%m%d')}.xlsx"
    response.headers["Content-Disposition"] = f"attachment; filename={filename}"
    return response


def _send_report_email(records, cfg, report_date, send_cc=True):
    """Send the gate report email via SMTP.

    Args:
        records: list of report record dicts
        cfg: reports_config dict
        report_date: YYYY-MM-DD string
        send_cc: if True, include CC recipients; if False, send to manager only

    Returns: (ok, msg_or_error)
    """
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders

    manager_email = cfg.get("manager_email", "")
    cc_emails_raw = cfg.get("cc_emails", "")
    cc_list = [e.strip() for e in cc_emails_raw.replace(";", ",").split(",") if e.strip()] if cc_emails_raw else []

    smtp_host = os.getenv("SMTP_HOST", "")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_pass = os.getenv("SMTP_PASSWORD", "")
    smtp_from = os.getenv("SMTP_FROM", smtp_user)

    if not smtp_host:
        return False, "SMTP_HOST environment variable not set"
    if not manager_email:
        return False, "No manager email configured"

    work_start = cfg.get("work_start", "08:00")
    late_threshold = int(cfg.get("late_threshold_minutes", 15))

    arrival_cam = cfg.get("arrival_camera", "")
    exit_cam = cfg.get("exit_camera", "")
    loc = db.get_location_by_source(arrival_cam or exit_cam)
    camera_name = loc["name"] if loc else (arrival_cam or exit_cam)

    xlsx_bytes = _build_report_excel(records, report_date, camera_name, work_start, late_threshold)

    late_count = sum(1 for r in records if r["arrived_late"] or r["late_exits_count"] > 0)
    html_body = f"""<html dir="rtl"><body style="font-family:Arial,sans-serif;color:#111;padding:24px;direction:rtl">
    <h2 style="margin:0 0 8px">تقرير البوابة اليومي — {report_date}</h2>
    <p style="color:#6b7280;margin:0 0 16px">الكاميرا: {camera_name} &nbsp;|&nbsp; بداية الدوام: {work_start} &nbsp;|&nbsp; حد التأخير: {late_threshold} دقيقة</p>
    <p style="font-size:15px">الإجمالي: <strong>{len(records)}</strong> موظف &nbsp;|&nbsp; مخالفات: <strong style="color:#dc2626">{late_count}</strong></p>
    <p style="color:#6b7280;font-size:13px">يرجى الاطلاع على الملف المرفق للتفاصيل الكاملة.</p>
    </body></html>"""

    recipients = [manager_email]
    if send_cc:
        recipients += cc_list

    msg = MIMEMultipart("mixed")
    msg["Subject"] = f"تقرير البوابة اليومي — {report_date}"
    msg["From"] = smtp_from
    msg["To"] = manager_email
    if send_cc and cc_list:
        msg["Cc"] = ", ".join(cc_list)
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    part = MIMEBase("application", "vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    part.set_payload(xlsx_bytes)
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", f'attachment; filename="gate_report_{report_date}.xlsx"')
    msg.attach(part)

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
            server.ehlo()
            if smtp_port != 25:
                server.starttls()
            if smtp_user and smtp_pass:
                server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_from, recipients, msg.as_string())
        return True, f"sent to {manager_email}" + (f" (CC: {', '.join(cc_list)})" if send_cc and cc_list else "")
    except Exception as e:
        log.error("Failed to send gate report email: %s", e)
        return False, str(e)


@app.route("/api/reports/send", methods=["POST"])
def api_reports_send():
    """Send daily gate report email. ?to=manager sends to manager only; ?to=all (default) sends to manager + CC."""
    if not db.is_available():
        return jsonify({"ok": False, "error": "database not available"}), 503

    cfg = _load_reports_config()
    arrival_cam = cfg.get("arrival_camera", "")
    exit_cam = cfg.get("exit_camera", "")
    if not arrival_cam and not exit_cam:
        return jsonify({"ok": False, "error": "No cameras configured"}), 400

    send_cc = request.args.get("to", "all") != "manager"

    late_threshold = int(cfg.get("late_threshold_minutes", 15))
    work_start = cfg.get("work_start", "08:00")
    night_shift_enabled = bool(cfg.get("night_shift_enabled", False))
    night_work_start = cfg.get("night_work_start", "21:00")
    night_late_threshold = int(cfg.get("night_late_threshold_minutes", 15))

    date_to = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=1)

    records = _generate_gate_report(arrival_cam, exit_cam, date_from, date_to, late_threshold, work_start,
                                    night_shift_enabled, night_work_start, night_late_threshold)
    report_date = datetime.now().strftime("%Y-%m-%d")

    ok, result = _send_report_email(records, cfg, report_date, send_cc=send_cc)
    if ok:
        return jsonify({"ok": True, "sent_to": result, "records": len(records)})
    return jsonify({"ok": False, "error": result}), 500


@app.route("/api/reports/history", methods=["GET"])
def api_reports_history():
    """List saved daily report summaries."""
    if not db.is_available():
        return jsonify({"ok": False, "error": "database not available"}), 503
    limit = min(int(request.args.get("limit", 60)), 365)
    rows = db.list_daily_reports(limit)
    return jsonify({"ok": True, "reports": rows})


@app.route("/api/reports/history/<date>", methods=["GET"])
def api_reports_history_date(date):
    """Return full saved report for a specific date."""
    if not db.is_available():
        return jsonify({"ok": False, "error": "database not available"}), 503
    row = db.get_daily_report(date)
    if not row:
        return jsonify({"ok": False, "error": "No saved report for this date"}), 404
    return jsonify({"ok": True, "report": row})


@app.route("/api/reports/history/<date>/save", methods=["POST"])
def api_reports_history_save(date):
    """Manually trigger save of report for a given date."""
    if not db.is_available():
        return jsonify({"ok": False, "error": "database not available"}), 503
    cfg = _load_reports_config()
    arrival_cam = cfg.get("arrival_camera", "")
    exit_cam = cfg.get("exit_camera", "")
    if not arrival_cam and not exit_cam:
        return jsonify({"ok": False, "error": "No cameras configured"}), 400
    try:
        date_from = datetime.strptime(date, "%Y-%m-%d").astimezone(timezone.utc)
    except ValueError:
        return jsonify({"ok": False, "error": "invalid date"}), 400
    date_to = date_from + timedelta(days=1)
    late_threshold = int(cfg.get("late_threshold_minutes", 15))
    work_start = cfg.get("work_start", "08:00")
    night_shift_enabled = bool(cfg.get("night_shift_enabled", False))
    night_work_start = cfg.get("night_work_start", "21:00")
    night_late_threshold = int(cfg.get("night_late_threshold_minutes", 15))
    records = _generate_gate_report(arrival_cam, exit_cam, date_from, date_to, late_threshold, work_start,
                                    night_shift_enabled, night_work_start, night_late_threshold)
    db.save_daily_report(date, arrival_cam, exit_cam, work_start, late_threshold, records)
    return jsonify({"ok": True, "date": date, "total": len(records)})


@app.route("/api/reports/history/<date>/export", methods=["GET"])
def api_reports_history_export(date):
    """Download Excel for a saved historical report."""
    if not db.is_available():
        return jsonify({"ok": False, "error": "database not available"}), 503
    row = db.get_daily_report(date)
    if not row:
        return jsonify({"ok": False, "error": "No saved report for this date"}), 404
    records = row.get("records", [])
    cfg = _load_reports_config()
    arrival_cam = row.get("arrival_camera") or cfg.get("arrival_camera", "")
    loc = db.get_location_by_source(arrival_cam)
    camera_name = loc["name"] if loc else arrival_cam
    xlsx_bytes = _build_report_excel(records, date, camera_name,
                                     row.get("work_start", "08:00"),
                                     row.get("late_threshold", 15))
    response = app.make_response(xlsx_bytes)
    response.headers["Content-Type"] = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    response.headers["Content-Disposition"] = f'attachment; filename="report_{date}.xlsx"'
    return response


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
    host, port = "0.0.0.0", 5001
    try:
        from waitress import serve  # type: ignore
        log.info("Serving with waitress on %s:%s", host, port)
        # Each MJPEG stream and SSE connection holds a thread for its lifetime.
        # With N cameras + M browser tabs open, 32 threads fills fast.
        # 100 gives headroom: 10 streams + 10 SSE + 80 regular requests concurrent.
        serve(app, host=host, port=port, threads=100, channel_timeout=300)
    except ImportError:
        log.warning("waitress not installed — falling back to Werkzeug dev server. "
                    "pip install waitress for production-grade concurrency.")
        app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)
