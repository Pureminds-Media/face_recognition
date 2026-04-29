"""Engine subprocess entrypoint.

Owns a single ``FaceEngine`` instance, runs the command loop on the parent
pipe, and continuously mirrors read-only state into a shared dict so the
Flask process can read fields without a roundtrip.

Spawned (not forked) by ``app.py`` at boot so the parent doesn't pay for
the heavy DeepFace / OpenCV imports and so CUDA contexts are clean.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid


def run(conn, state, engine_kwargs, log_level=logging.INFO):
    logging.basicConfig(
        level=log_level,
        format="[engine] %(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("engine_runner")

    # Imports happen in the child so the parent stays light.
    from face_engine import FaceEngine
    from dotenv import load_dotenv
    load_dotenv()

    os.makedirs(engine_kwargs.get("known_dir", "faces"), exist_ok=True)
    engine = FaceEngine(**engine_kwargs)

    # Apply saved grid/cam config so the child mirrors what the user had.
    saved = FaceEngine.load_grid_config()
    if saved is not None:
        try:
            engine.set_grid_layout(*saved["layout"])
            engine.cam_index = saved["cam_index"]
        except (ValueError, KeyError):
            pass

    state["identities"] = len(engine.known_embeddings)
    state["identity_names"] = [name for name, _ in engine.known_embeddings]

    # ---- Background thread: mirror engine state into the shared dict ----
    stop_evt = threading.Event()

    def _mirror_loop():
        while not stop_evt.is_set():
            try:
                state["running"] = bool(engine.is_running())
                state["cam_index"] = engine.cam_index
                state["viewer_source"] = engine.viewer_source
                state["viewer_mode"] = engine.viewer_mode
                state["viewer_grid_offset"] = engine.viewer_grid_offset
                state["_grid_layout"] = engine._grid_layout
                state["is_grid_mode"] = engine._is_grid_mode()
                state["grid_page_size"] = engine.grid_page_size()
                state["grid_page_count"] = engine.grid_page_count()
                state["activity_enabled"] = bool(getattr(engine, "activity_enabled", False))
                state["source_name_map"] = dict(getattr(engine, "source_name_map", {}) or {})

                # Hot-path data
                state["tracks"] = list(engine.get_tracks() or [])
                state["jpeg"] = engine.get_jpeg()
                state["qr_state"] = tuple(engine.get_qr_state() or (None, 0.0))

                # Activity tables (engine internals; tolerate absence)
                act_by_name = {}
                act_by_visit = {}
                try:
                    for nm in list(getattr(engine, "_activity_results", {}).keys()):
                        act_by_name[nm] = tuple(engine.get_activity(nm))
                except Exception: pass
                try:
                    for vid in list(getattr(engine, "_activity_counts", {}).keys()):
                        v = engine.get_visit_top_activity(vid)
                        if v is not None:
                            act_by_visit[vid] = v
                except Exception: pass
                state["activity_by_name"] = act_by_name
                state["activity_by_visit"] = act_by_visit

                state["identities"] = len(engine.known_embeddings)
                state["identity_names"] = [name for name, _ in engine.known_embeddings]
            except Exception as e:
                log.warning("mirror loop error: %s", e)
            time.sleep(0.05)

    threading.Thread(target=_mirror_loop, daemon=True).start()

    # ---- Test job runner (process_video_file) ----
    def _run_test_job(job_id, input_path, output_path):
        jobs = dict(state.get("test_jobs", {}) or {})
        jobs[job_id] = {"progress": 0.0, "done": False}
        state["test_jobs"] = jobs

        def _progress(p):
            try:
                jobs = dict(state.get("test_jobs", {}) or {})
                cur = jobs.get(job_id) or {}
                cur["progress"] = float(p or 0.0)
                jobs[job_id] = cur
                state["test_jobs"] = jobs
            except Exception: pass

        try:
            engine.reload_faces()
            result = engine.process_video_file(input_path, output_path, progress_cb=_progress)
            jobs = dict(state.get("test_jobs", {}) or {})
            jobs[job_id] = {"progress": 1.0, "done": True, "result": result}
            state["test_jobs"] = jobs
        except Exception as e:
            jobs = dict(state.get("test_jobs", {}) or {})
            jobs[job_id] = {"progress": last_progress(jobs.get(job_id, {})),
                            "done": True, "error": str(e)}
            state["test_jobs"] = jobs

    def last_progress(d):
        try: return float(d.get("progress") or 0.0)
        except Exception: return 0.0

    # ---- Command dispatch ----
    def _dispatch(op, args, kwargs):
        if op == "setattr":
            name, value = args
            setattr(engine, name, value)
            # Mirror immediately so the parent sees the new value before
            # the periodic sync thread next ticks.
            try: state[name] = value
            except Exception: pass
            return None
        if op == "process_video_start":
            input_path, output_path = args
            job_id = uuid.uuid4().hex
            threading.Thread(
                target=_run_test_job, args=(job_id, input_path, output_path), daemon=True,
            ).start()
            return job_id
        # Map op name → engine method.
        method = getattr(engine, op, None)
        if method is None or not callable(method):
            raise AttributeError(f"unknown command {op!r}")
        return method(*args, **kwargs)

    log.info("engine_runner ready (cam_index=%s, identities=%s)",
             engine.cam_index, state.get("identities"))

    try:
        while True:
            try:
                msg = conn.recv()
            except EOFError:
                break
            if msg == "shutdown":
                break
            try:
                op, args, kwargs = msg
                result = _dispatch(op, args, kwargs)
                conn.send((True, result))
            except Exception as e:
                log.exception("command failed: %s", msg)
                conn.send((False, e))
    finally:
        stop_evt.set()
        try: engine.stop()
        except Exception: pass
        log.info("engine_runner exiting")
