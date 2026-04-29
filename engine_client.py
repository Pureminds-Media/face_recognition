"""Thin proxy that exposes a FaceEngine-compatible surface to the Flask
process while the actual engine runs in a child process.

Architecture
------------
- Commands (start/stop/reload/set_*/footage/test job) → pickled tuple sent
  over a multiprocessing.Pipe; the child applies the call and replies.
- Fast reads (is_running, tracks, jpeg, activity tables, qr state, viewer
  fields, etc.) → multiprocessing.Manager().dict() that the child keeps
  fresh. No pipe roundtrip per read.
- Attribute writes (e.g. ``engine.cam_index = …``) → forwarded as a
  ``setattr`` command so the child holds the source of truth.

Hot path note: ``get_jpeg()`` and ``get_tracks()`` are read from the
shared state dict on every call. Manager-backed reads are ~100 µs each;
fine for the current frame rate. If we ever push to higher fps this can
move to a shared_memory ring buffer without changing the public API.
"""

from __future__ import annotations

import threading
import time
import uuid


class EngineClient:
    """Proxy for the engine subprocess. Public surface mirrors FaceEngine."""

    def __init__(self, conn, state, lock):
        # Use object.__setattr__ to bypass our overridden __setattr__.
        object.__setattr__(self, "_conn", conn)
        object.__setattr__(self, "_state", state)
        object.__setattr__(self, "_lock", lock)

    # ---------- low-level ----------
    def _call(self, op, *args, **kwargs):
        """Send a command and block until the engine acknowledges."""
        with self._lock:
            self._conn.send((op, args, kwargs))
            ok, val = self._conn.recv()
        if not ok:
            raise val
        return val

    # ---------- lifecycle ----------
    def start(self): return self._call("start")
    def stop(self): return self._call("stop")

    # ---------- viewer / grid ----------
    def set_grid_layout(self, rows, cols): return self._call("set_grid_layout", rows, cols)
    def set_viewer(self, **kwargs): return self._call("set_viewer", **kwargs)
    def grid_page_size(self): return int(self._state.get("grid_page_size", 1))
    def grid_page_count(self): return int(self._state.get("grid_page_count", 1))
    def _is_grid_mode(self): return bool(self._state.get("is_grid_mode", False))

    @staticmethod
    def _parse_grid_sources(cam_index):
        # Static helper that doesn't need the engine process — just delegate.
        from face_engine import FaceEngine
        return FaceEngine._parse_grid_sources(cam_index)

    # ---------- recognition / faces ----------
    def reload_faces(self): return self._call("reload_faces")

    @property
    def known_embeddings(self):
        # Routes use this in two ways: ``len(...)`` for a count, and
        # ``for name, _ in known_embeddings`` to iterate names. We don't
        # ship raw embedding vectors over IPC (they'd serialise on every
        # access); instead expose ``(name, None)`` pairs.
        return [(n, None) for n in (self._state.get("identity_names") or [])]

    # ---------- footage ----------
    def start_footage(self, *args, **kwargs): return self._call("start_footage", *args, **kwargs)
    def stop_footage(self, visit_id): return self._call("stop_footage", visit_id)
    def stop_all_footage(self): return self._call("stop_all_footage")

    # ---------- activity ----------
    def get_activity(self, name):
        v = self._state.get("activity_by_name", {}).get(name)
        return tuple(v) if v else (None, 0.0)

    def record_activity_for_visit(self, vid, label):
        return self._call("record_activity_for_visit", vid, label)

    def get_visit_top_activity(self, vid):
        return self._state.get("activity_by_visit", {}).get(vid)

    def clear_visit_activity(self, vid):
        return self._call("clear_visit_activity", vid)

    # ---------- frames / tracks ----------
    def is_running(self): return bool(self._state.get("running", False))
    def get_jpeg(self): return self._state.get("jpeg")
    def get_tracks(self): return list(self._state.get("tracks", []))
    def get_qr_state(self):
        v = self._state.get("qr_state")
        return tuple(v) if v else (None, 0.0)

    def get_snapshot(self, person_name, camera_source=None):
        # Pulled live from the engine because snapshots are write-once /
        # read-once and we don't want to keep them in shared state.
        return self._call("get_snapshot", person_name, camera_source)

    # ---------- offline test job ----------
    def process_video_file(self, input_path, output_path, progress_cb=None):
        """Run a full test pipeline in the engine process.

        The engine pushes progress through ``state['test_jobs'][job_id]``;
        this method polls until done so the caller's ``progress_cb`` keeps
        firing exactly like before.
        """
        job_id = self._call("process_video_start", input_path, output_path)
        last_progress = 0.0
        while True:
            time.sleep(0.15)
            jobs = self._state.get("test_jobs", {}) or {}
            st = jobs.get(job_id)
            if not st:
                continue
            p = float(st.get("progress") or 0.0)
            if progress_cb and p > last_progress:
                last_progress = p
                try: progress_cb(p)
                except Exception: pass
            if st.get("done"):
                # Clear so memory doesn't grow.
                try:
                    jobs = dict(self._state.get("test_jobs", {}) or {})
                    jobs.pop(job_id, None)
                    self._state["test_jobs"] = jobs
                except Exception: pass
                if st.get("error"):
                    raise RuntimeError(st["error"])
                return st.get("result") or {}

    # ---------- generic attribute proxy ----------
    # Keys synced into the shared state by the engine runner; reading these
    # is a Manager dict get, no pipe roundtrip.
    _MIRRORED = {
        "cam_index", "viewer_source", "viewer_mode", "viewer_grid_offset",
        "_grid_layout", "activity_enabled", "source_name_map",
    }

    def __getattr__(self, name):
        # Block client-internal names from leaking into the proxy lookup
        # (state, conn, lock) so we don't recurse forever.
        if name in ("_state", "_conn", "_lock"):
            raise AttributeError(name)
        st = object.__getattribute__(self, "_state")
        if name in st:
            return st[name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name.startswith("_") and name not in self._MIRRORED:
            object.__setattr__(self, name, value)
            return
        self._call("setattr", name, value)
