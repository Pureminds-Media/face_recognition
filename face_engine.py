import os
import json
import re
import time
import struct
import threading
import collections
from queue import Queue, Empty, Full
import cv2
import numpy as np
from deepface import DeepFace
from action_detector import ActionDetector
from head_detector import HeadDetector

GRID_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grid_config.json")
AVAILABLE_LAYOUTS = [(2, 2), (3, 3), (4, 4)]


def l2norm(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-10)


def cosine_distance(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return 1.0 - float(np.dot(a, b) / denom)


def _mp4_faststart(path):
    """Move the moov atom before mdat so browsers can stream the MP4.

    Rewrites the file in-place.  If anything goes wrong, the original file
    is left untouched.  Only operates on .mp4 files.
    """
    if not path.lower().endswith(".mp4"):
        return
    try:
        with open(path, "rb") as f:
            data = f.read()
        # Parse top-level boxes to find moov and mdat
        boxes = []
        pos = 0
        while pos < len(data):
            if pos + 8 > len(data):
                break
            size = struct.unpack(">I", data[pos : pos + 4])[0]
            box_type = data[pos + 4 : pos + 8]
            if size == 0:
                break
            if size == 1:
                if pos + 16 > len(data):
                    break
                size = struct.unpack(">Q", data[pos + 8 : pos + 16])[0]
            boxes.append((box_type, pos, size))
            pos += size

        moov_idx = None
        mdat_idx = None
        for i, (bt, bp, bs) in enumerate(boxes):
            if bt == b"moov":
                moov_idx = i
            if bt == b"mdat":
                mdat_idx = i

        if moov_idx is None or mdat_idx is None:
            return  # not a standard mp4
        if moov_idx < mdat_idx:
            return  # already fast-started

        # Rebuild: everything before mdat + moov + mdat + everything after moov
        moov_type, moov_pos, moov_size = boxes[moov_idx]
        moov_data = data[moov_pos : moov_pos + moov_size]

        # We need to adjust chunk offsets inside moov by the shift amount
        # (moov is being moved before mdat, so mdat shifts forward by moov_size)
        mdat_type, mdat_pos, mdat_size = boxes[mdat_idx]
        shift = moov_size  # moov is inserted before mdat, pushing mdat forward

        # Adjust stco (32-bit) and co64 (64-bit) chunk offsets in moov
        adjusted_moov = bytearray(moov_data)
        _adjust_offsets(adjusted_moov, shift)

        # Rebuild the file
        before_mdat = data[:mdat_pos]
        mdat_and_after = data[mdat_pos:moov_pos]
        after_moov = data[moov_pos + moov_size:]

        new_data = before_mdat + bytes(adjusted_moov) + mdat_and_after + after_moov

        tmp_path = path + ".tmp"
        with open(tmp_path, "wb") as f:
            f.write(new_data)
        os.replace(tmp_path, path)
    except Exception:
        # If anything goes wrong, leave the original file
        try:
            if os.path.exists(path + ".tmp"):
                os.remove(path + ".tmp")
        except Exception:
            pass


def _adjust_offsets(moov_data, shift):
    """Adjust stco/co64 chunk offsets inside a moov box by `shift` bytes."""
    pos = 8  # skip moov header
    _adjust_offsets_recursive(moov_data, pos, len(moov_data), shift)


def _adjust_offsets_recursive(data, start, end, shift):
    """Walk nested boxes inside moov and adjust stco/co64 entries."""
    pos = start
    while pos < end - 8:
        size = struct.unpack(">I", data[pos : pos + 4])[0]
        box_type = bytes(data[pos + 4 : pos + 8])
        if size < 8 or pos + size > end:
            break
        if box_type == b"stco":
            # 32-bit chunk offset table
            entry_count = struct.unpack(">I", data[pos + 12 : pos + 16])[0]
            for i in range(entry_count):
                off = pos + 16 + i * 4
                old = struct.unpack(">I", data[off : off + 4])[0]
                struct.pack_into(">I", data, off, old + shift)
        elif box_type == b"co64":
            # 64-bit chunk offset table
            entry_count = struct.unpack(">I", data[pos + 12 : pos + 16])[0]
            for i in range(entry_count):
                off = pos + 16 + i * 8
                old = struct.unpack(">Q", data[off : off + 8])[0]
                struct.pack_into(">Q", data, off, old + shift)
        elif box_type in (b"trak", b"mdia", b"minf", b"stbl"):
            # Recurse into container boxes
            _adjust_offsets_recursive(data, pos + 8, pos + size, shift)
        pos += size


def iou(a, b):
    ax, ay, aw, ah = map(float, a)
    bx, by, bw, bh = map(float, b)
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (aw * ah) + (bw * bh) - inter + 1e-10
    return inter / union


def create_tracker(tracker_type: str):
    tracker_type = tracker_type.upper()

    def _try(factory_name: str):
        if hasattr(cv2, factory_name):
            return getattr(cv2, factory_name)()
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, factory_name):
            return getattr(cv2.legacy, factory_name)()
        return None

    if tracker_type == "CSRT":
        tr = _try("TrackerCSRT_create")
    else:
        tr = _try("TrackerKCF_create")

    if tr is None:
        raise RuntimeError(
            "OpenCV tracker not available. Install opencv-contrib-python "
            "(and uninstall opencv-python if both are installed)."
        )
    return tr


class FaceEngine:
    def __init__(
        self,
        known_dir="faces",
        detector="retinaface",
        model="Facenet512",
        threshold=0.4,
        detect_every=1.0,
        detect_scale=1.0,
        tracker_type="CSRT",
        cam_index=0,
        width=1920,
        height=1080,
        out_fps=15,
        jpeg_quality=80,
    ):
        self.known_dir = known_dir
        self.detector = detector
        self.model = model
        self.threshold = threshold
        self.detect_every = detect_every
        self.detect_scale = detect_scale
        self.tracker_type = tracker_type
        self.cam_index = cam_index
        self.width = width
        self.height = height
        self.out_fps = out_fps
        self.jpeg_quality = jpeg_quality
        self.allowed_exts = (".jpg", ".png", ".jpeg", ".webp")
        self.known_embeddings = []
        self.tracks = []
        self.in_q = Queue(maxsize=1)
        self.out_q = Queue(maxsize=1)
        self.stop_evt = threading.Event()
        self._cap = None
        self._grid_workers = {}
        self._grid_sources = []
        # Optional friendly-name lookup for grid tile labels:
        # { "<source-string>": "<friendly name>" }
        # Set by callers (e.g. app.py) before/after start(). Falls back to
        # "Cam <source>" when a source isn't in the map.
        self.source_name_map = {}
        self._grid_stop_evt = threading.Event()
        self._grid_render_t = None
        self._grid_layout = (2, 2)  # rows, cols  (may be overridden by load_grid_config)
        self._grid_qr_rr_idx = 0
        self._worker_t = None
        self._main_t = None
        self._lock = threading.Lock()
        # Serialize all DeepFace inference calls. With many simultaneous
        # detect_loops the underlying TensorFlow models race during lazy
        # model load / state mutation; the symptom is that *no* boxes
        # ever appear because every call either crashes silently or
        # returns garbage. The lock costs us parallelism but the GPU
        # was already the bottleneck anyway.
        self._inference_lock = threading.Lock()
        self._latest_jpeg = None
        # Per-camera annotated full-resolution JPEG for the source the
        # user is currently watching in single mode. None when no source
        # is selected or the source is offline.
        self._latest_single_jpeg = None
        # Viewer state (decoupled from the analysis pool). The engine
        # analyses every camera; viewer_* just controls what get_jpeg()
        # returns to the MJPEG stream.
        self.viewer_mode = "grid"   # "single" | "grid"
        self.viewer_source = ""     # str representation of the chosen source
        # In grid mode, this is the index of the first camera shown in
        # the visible composite. Lets the user page through more cameras
        # than the layout has slots for.
        self.viewer_grid_offset = 0
        self._latest_tracks = []
        self._running = False
        self._qr_detector = cv2.QRCodeDetector()
        self.qr_scan_every = 0.5  # seconds (2 scans/sec). Increase if CPU is high.
        self._last_qr_scan_t = 0.0
        self._latest_qr = None  # last decoded QR string
        self._latest_qr_t = 0.0 # monotonic time of last decoded QR
        # Snapshot buffer: (person_name, camera_source) -> JPEG bytes
        # Holds the latest annotated tile/frame for each person/camera pair.
        # Consumed by app.py when opening a new visit.
        self._snapshot_buf = {}
        self._snapshot_lock = threading.Lock()

        # Rolling frame ring buffer per camera source (used for snapshots).
        # Key: str(camera_source)  Value: deque of numpy_frame
        self.FOOTAGE_RING_SECS = 5.0  # keep a few seconds for snapshot quality
        self._frame_ring = {}  # camera_source -> deque
        self._frame_ring_lock = threading.Lock()
        self._ring_frame_counts = {}  # camera_source -> (start_mono, frame_count)
        # Active footage writers: visit_id -> {cam, person, writer, ...}
        # Each active visit has its own cv2.VideoWriter streaming frames to disk.
        self._active_writers = {}
        self._writers_lock = threading.Lock()

        # Head detection (YOLO) — supplements face detector for track persistence
        self._head_detector = None  # lazy, created in start()

        # Action detection (CLIP zero-shot) — can be disabled via env.
        # Default OFF: on large deployments (10+ cams) CLIP dominates GPU
        # and starves the face/head detectors. Set ACTION_DETECTION_ENABLED=true
        # to opt back in.
        self.activity_enabled = os.getenv("ACTION_DETECTION_ENABLED", "false").lower() in ("true", "1", "yes")
        self._action_detector = None
        self.activity_detect_every = 2.0  # seconds between activity checks per person
        # Dedicated action detection thread: runs async, never blocks detect/render loops
        self._activity_q = Queue(maxsize=32)
        self._activity_results = {}   # person_id -> {"label": str, "conf": float, "t": float}
        self._activity_results_lock = threading.Lock()
        self._activity_t = None
        # Per-visit activity tallies: visit_id -> Counter of label strings
        self._activity_counts = {}
        self._activity_counts_lock = threading.Lock()

        # Auto-capture unknowns — disabled by default (not suited for large crowds)
        self.auto_capture_enabled = os.getenv("AUTO_CAPTURE_ENABLED", "false").lower() in ("true", "1", "yes")
        print(f"[face-engine] auto_capture_enabled={self.auto_capture_enabled} activity_enabled={self.activity_enabled}")
        self._unknown_pending = {}  # id(track) -> {"first_t", "best_frame", "best_bbox", "best_area"}
        # Save an unknown after they've been tracked continuously for this
        # many seconds. Time-based (not detection-count-based) so cadence
        # changes / lock contention don't affect when capture fires.
        self._unknown_capture_min_seconds = 5.0
        self._unknown_max_auto = 150  # max unknown_N folders
        # After an unknown_N is created, keep adding sample images while
        # the same track is still in frame, up to this cap. More samples
        # = more diverse embeddings = better recognition next time the
        # person reappears.
        self._unknown_max_images_per_person = 30
        self._unknown_extra_capture_min_interval = 1.0  # seconds between extra saves
        # Per-(person, camera_source) cooldown for extra captures. Saving a
        # fresh sample on the *same* camera is rate-limited (default 10 min)
        # since consecutive frames look near-identical, but a different
        # camera produces genuinely new viewpoints/lighting and is allowed
        # immediately. Applies to both unknown_N folders and human-named
        # people once enrolled.
        self._extra_capture_per_cam_cooldown = 600.0
        self._last_extra_capture_at = {}  # (name, camera_source) -> monotonic ts
        self._last_extra_capture_lock = threading.Lock()

    # ---------- public ----------
    def start(self):
        if self._running:
            return

        self.stop_evt.clear()
        self.reload_faces()

        if self._is_grid_mode():
            self._start_grid_mode()
        else:
            cap = cv2.VideoCapture(self.cam_index)
            if not cap.isOpened():
                raise RuntimeError("Cannot open webcam")

            # Use the camera's native resolution — don't constrain it.
            # Minimize internal buffer to reduce lag (matches grid mode)
            if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
            # Drain stale buffered frames so first displayed frame is current
            for _ in range(5):
                cap.grab()
            self._cap = cap

            self._worker_t = threading.Thread(target=self._detector_worker, daemon=True)
            self._worker_t.start()

            self._main_t = threading.Thread(target=self._main_loop, daemon=True)
            self._main_t.start()

        # Start head detection (always on — supplements face detector)
        try:
            self._head_detector = HeadDetector()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("HeadDetector not available: %s", e)
            self._head_detector = None

        # Start action detection thread (shared by grid and single-cam modes)
        if self.activity_enabled:
            self._action_detector = ActionDetector()
            self._activity_t = threading.Thread(target=self._activity_loop, daemon=True)
            self._activity_t.start()

        self._running = True

    def stop(self):
        if not self._running:
            return

        import logging
        _log = logging.getLogger(__name__)

        # Signal threads to exit. Join BEFORE releasing self._cap, since
        # _main_loop calls self._cap.read() — releasing while it's blocked
        # inside read() (e.g. on RTSP) is a use-after-free → segfault.
        self._running = False
        self.stop_evt.set()

        # Tear down grid workers (their own cleanup joins+releases safely)
        self._cleanup_grid_mode()

        # Join threads. Use a timeout > FFmpeg stimeout (5s) so RTSP
        # workers actually have a chance to return from a blocking read.
        worker_alive = False
        main_alive = False
        if self._worker_t is not None:
            self._worker_t.join(timeout=8.0)
            if self._worker_t.is_alive():
                worker_alive = True
                _log.warning("stop(): _worker_t did not exit within timeout — zombie thread")
            self._worker_t = None

        if self._main_t is not None:
            self._main_t.join(timeout=8.0)
            if self._main_t.is_alive():
                main_alive = True
                _log.warning("stop(): _main_t did not exit within timeout — zombie thread")
            self._main_t = None

        # Release the capture only after the readers are gone. If _main_t
        # is a zombie still holding a reference, leak the cap rather than
        # crash; the OS reclaims it on process exit.
        if self._cap is not None:
            if not main_alive:
                try:
                    self._cap.release()
                except Exception:
                    pass
            else:
                _log.warning("stop(): leaking _cap because _main_t is still alive")
            self._cap = None

        # Stop activity thread
        if self._activity_t is not None:
            # Drain first so sentinel can be inserted
            while not self._activity_q.empty():
                try:
                    self._activity_q.get_nowait()
                except Empty:
                    break
            try:
                self._activity_q.put_nowait(None)  # sentinel
            except Full:
                pass
            self._activity_t.join(timeout=3.0)
            if self._activity_t.is_alive():
                _log.warning("stop(): _activity_t did not exit within timeout — zombie thread")
            self._activity_t = None
        # Drain any remaining activity queue items
        while not self._activity_q.empty():
            try:
                self._activity_q.get_nowait()
            except Empty:
                break
        with self._activity_results_lock:
            self._activity_results.clear()
        with self._activity_counts_lock:
            self._activity_counts.clear()

        # Release action detector (free GPU memory — CLIP model)
        self._action_detector = None

        # Release head detector (free GPU memory — YOLO model)
        self._head_detector = None

        # Close all active footage writers (release file handles + codec resources)
        try:
            self.stop_all_footage()
        except Exception as e:
            _log.warning("stop(): stop_all_footage error: %s", e)

        # Clear frame ring buffers (can be hundreds of MB of numpy arrays)
        with self._frame_ring_lock:
            self._frame_ring.clear()
            self._ring_frame_counts.clear()

        # Clear snapshot buffer
        with self._snapshot_lock:
            self._snapshot_buf.clear()

        # Clear unknown pending accumulator
        self._unknown_pending.clear()

        # Clear queues
        with self.in_q.mutex:
            self.in_q.queue.clear()
        with self.out_q.mutex:
            self.out_q.queue.clear()

        # Clear published state
        with self._lock:
            self._latest_jpeg = None
            self._latest_tracks = []
            self._latest_qr = None
            self._latest_qr_t = 0.0

        self.tracks = []

    # ---------- action detection thread ----------

    def _activity_loop(self):
        """Dedicated thread for CLIP action detection.

        Pulls (frame, bbox, person_id) items from the queue and writes
        results to self._activity_results.  Runs fully async so it never
        blocks the detection or render loops.
        """
        while True:
            try:
                item = self._activity_q.get(timeout=1.0)
            except Empty:
                if self.stop_evt.is_set():
                    break
                continue
            if item is None:          # sentinel → exit
                break
            frame, bbox, person_id = item

            # Drain any older items for the same person (keep only latest)
            latest_frame, latest_bbox = frame, bbox
            while not self._activity_q.empty():
                try:
                    peek = self._activity_q.get_nowait()
                except Empty:
                    break
                if peek is None:      # sentinel
                    self._activity_q.put(None)
                    break
                pf, pb, pid = peek
                if pid == person_id:
                    latest_frame, latest_bbox = pf, pb
                else:
                    # Different person — put it back and stop draining
                    try:
                        self._activity_q.put_nowait(peek)
                    except Full:
                        pass
                    break
            frame, bbox = latest_frame, latest_bbox

            try:
                label, conf = self._action_detector.detect(
                    frame, bbox, person_id=person_id
                )
                with self._activity_results_lock:
                    self._activity_results[person_id] = {
                        "label": label,
                        "conf": conf,
                        "t": time.monotonic(),
                    }
            except Exception:
                pass

    def get_activity(self, person_id):
        """Return (label, conf) for a person, or (None, 0.0).

        Results older than activity_detect_every * 2 are treated as stale
        (e.g. person turned around — face detector no longer confirming).
        """
        if not self.activity_enabled:
            return (None, 0.0)
        with self._activity_results_lock:
            r = self._activity_results.get(person_id)
        if r is None:
            return (None, 0.0)
        # Expire stale results (face no longer visible → stop showing action)
        if (time.monotonic() - float(r.get("t", 0.0))) > self.activity_detect_every * 2.0:
            return (None, 0.0)
        return (r["label"], r["conf"])

    def record_activity_for_visit(self, visit_id, label):
        """Increment the activity tally for an active visit."""
        if label is None:
            return
        with self._activity_counts_lock:
            if visit_id not in self._activity_counts:
                self._activity_counts[visit_id] = collections.Counter()
            self._activity_counts[visit_id][label] += 1

    def get_visit_top_activity(self, visit_id):
        """Return the most frequent activity label for a visit, or None."""
        with self._activity_counts_lock:
            ctr = self._activity_counts.get(visit_id)
        if not ctr:
            return None
        most = ctr.most_common(1)
        return most[0][0] if most else None

    def clear_visit_activity(self, visit_id):
        """Remove activity tally for a closed visit."""
        with self._activity_counts_lock:
            self._activity_counts.pop(visit_id, None)

    def is_running(self):
        return self._running

    def get_jpeg(self):
        """Return the JPEG bytes appropriate for the current viewer mode.

        Single mode returns the annotated full-resolution frame for the
        chosen camera (or None until the first frame for that source has
        been rendered). Grid mode returns the composite. We deliberately
        do NOT fall back from single→grid: that produced a noticeable
        flash of the grid composite every time the user switched cameras.
        Returning None makes the MJPEG generator briefly hold the last
        frame in the browser instead.
        """
        with self._lock:
            if self.viewer_mode == "single":
                return self._latest_single_jpeg
            return self._latest_jpeg

    def set_viewer(self, mode=None, source=None, grid_offset=None):
        """Update viewer state. Cheap operation — does not touch the
        analysis pool. Cameras keep running regardless. We do NOT clear
        ``_latest_single_jpeg`` here; the next render produces a fresh
        frame for the new source on its own and a stale frame is far
        less jarring than a black gap or a grid flash."""
        if mode is not None and mode in ("single", "grid"):
            self.viewer_mode = mode
        if source is not None:
            self.viewer_source = str(source)
        if grid_offset is not None:
            self.viewer_grid_offset = max(0, int(grid_offset))

    def grid_page_count(self):
        """How many pages of cameras the current grid layout has."""
        rows, cols = self._grid_layout
        slots = max(1, rows * cols)
        # Use the union of grid_sources and worker keys (matches the
        # render loop's analysis_sources construction).
        n = len(self._grid_sources)
        for src in self._grid_workers.keys():
            if src not in self._grid_sources:
                n += 1
        return max(1, (n + slots - 1) // slots)

    def grid_page_size(self):
        rows, cols = self._grid_layout
        return max(1, rows * cols)

    def _render_single_no_signal(self, source):
        """Produce a placeholder annotated frame when the viewer's chosen
        camera is offline. Sized to ``self.width × self.height`` so the
        MJPEG stream stays at a consistent resolution."""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        msg = f"No Signal [{source}]"
        cv2.putText(
            frame, msg,
            (max(8, self.width // 2 - 200), self.height // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3,
        )
        ok, buf = cv2.imencode(
            ".jpg", frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)],
        )
        if ok:
            with self._lock:
                self._latest_single_jpeg = buf.tobytes()

    def get_tracks(self):
        with self._lock:
            return list(self._latest_tracks)
    
    def get_snapshot(self, person_name, camera_source=None):
        """Pop and return buffered JPEG bytes for a person/camera, or None.

        In grid mode, camera_source is required to identify which tile.
        In single-camera mode, camera_source can be None.
        """
        key = (person_name, str(camera_source) if camera_source is not None else None)
        with self._snapshot_lock:
            return self._snapshot_buf.pop(key, None)

    def _push_frame_to_ring(self, camera_source, frame):
        """Push an annotated frame into the rolling ring buffer for a camera.

        Throttled to out_fps via frame-count so timing is drift-free.
        """
        key = str(camera_source) if camera_source is not None else None
        now = time.monotonic()
        fps = max(1, self.out_fps)
        with self._frame_ring_lock:
            if key not in self._frame_ring:
                maxlen = int(fps * self.FOOTAGE_RING_SECS) + 10
                self._frame_ring[key] = collections.deque(maxlen=maxlen)
            # Frame-count throttle
            start_t, count = self._ring_frame_counts.get(key, (now, 0))
            if key not in self._ring_frame_counts:
                self._ring_frame_counts[key] = (now, 0)
                start_t, count = now, 0
            expected_time = count / fps
            elapsed = now - start_t
            if count > 0 and expected_time > elapsed:
                return  # too early for next frame
            self._frame_ring[key].append(frame.copy())
            self._ring_frame_counts[key] = (start_t, count + 1)

    def start_footage(self, visit_id, person_name, camera_source, footage_dir, fname):
        """Open a VideoWriter for a visit.

        Called by app.py when a new visit opens.  The writer stays open and
        receives frames from _feed_active_writers() only while the person is
        visible, until stop_footage() is called.

        Returns (ok: bool, actual_fname: str).
        """
        cam_key = str(camera_source) if camera_source is not None else None
        fpath = os.path.join(footage_dir, fname)
        fps = max(1, self.out_fps)

        # Determine frame dimensions — prefer ring buffer (native resolution),
        # then raw frame from worker, then full canvas dimensions.
        w, h = None, None
        with self._frame_ring_lock:
            ring = self._frame_ring.get(cam_key)
            if ring:
                h, w = ring[-1].shape[:2]
        if w is None and self._is_grid_mode():
            worker = self._grid_workers.get(camera_source)
            if worker is not None:
                with worker["lock"]:
                    raw = worker.get("latest_raw_frame")
                if raw is not None:
                    h, w = raw.shape[:2]
        if w is None:
            w, h = self.width, self.height

        # Use VP8/WebM — natively supported in all modern browsers.
        # OpenCV emits a misleading "tag VP80 is not supported" warning,
        # but the files are valid WebM with correct EBML headers.
        fname_webm = fname.rsplit(".", 1)[0] + ".webm"
        fpath = os.path.join(footage_dir, fname_webm)
        fname = fname_webm
        fourcc = cv2.VideoWriter_fourcc(*"VP80")
        writer = cv2.VideoWriter(fpath, fourcc, fps, (w, h))
        if not writer.isOpened():
            writer.release()
            return False, fname

        now = time.monotonic()
        with self._writers_lock:
            self._active_writers[visit_id] = {
                "cam": cam_key,
                "person": person_name,
                "writer": writer,
                "path": fpath,
                "fname": fname,
                "w": w,
                "h": h,
                "fps": fps,
                "start_time": now,
                "frame_count": 0,      # frames written so far (throttle pacing)
                "total_frames": 0,     # same as frame_count (no pause/resume)
            }
        return True, fname

    def _feed_active_writers(self, camera_source, frame, visible_persons):
        """Write the current frame to every active VideoWriter on this camera.

        Records continuously from visit start to visit end regardless of
        whether the person is currently detected — this avoids jumpy cuts
        when someone turns around or is briefly occluded.

        Uses a frame-count throttle with duplicate-frame padding so the
        video plays back at real-time speed even when the render loop is
        slower than the target FPS.
        """
        cam_key = str(camera_source) if camera_source is not None else None
        now = time.monotonic()
        with self._writers_lock:
            for vid, rec in list(self._active_writers.items()):
                if rec["cam"] != cam_key:
                    continue
                # Frame-count throttle: write enough frames so that
                # frame_count stays in sync with wall-clock time.
                fps = rec["fps"]
                elapsed = now - rec["start_time"]
                target_frames = int(elapsed * fps) + 1  # where we should be
                deficit = target_frames - rec["frame_count"]
                if deficit <= 0:
                    continue
                # Safety cap: never write more than 2× fps frames in one call
                # to avoid runaway writes if start_time is stale.
                deficit = min(deficit, fps * 2)
                try:
                    fh, fw = frame.shape[:2]
                    if (fw, fh) != (rec["w"], rec["h"]):
                        f = cv2.resize(frame, (rec["w"], rec["h"]))
                    else:
                        f = frame
                    for _ in range(deficit):
                        rec["writer"].write(f)
                    rec["frame_count"] += deficit
                    rec["total_frames"] += deficit
                except Exception:
                    pass  # writer may have been released

    def stop_footage(self, visit_id):
        """Close the VideoWriter for a visit.

        Returns (fname, visible_secs) or (None, 0).
        visible_secs = total_frames / fps — the exact on-camera duration
        across all visible segments (survives pause/resume cycles).
        """
        with self._writers_lock:
            rec = self._active_writers.pop(visit_id, None)
        if rec is None:
            return None, 0.0
        try:
            rec["writer"].release()
        except Exception:
            pass
        visible_secs = rec["total_frames"] / max(1, rec["fps"])
        return rec["fname"], visible_secs

    def get_footage_visible_secs(self, visit_id):
        """Return current visible seconds for an active writer, or 0."""
        with self._writers_lock:
            rec = self._active_writers.get(visit_id)
            if rec is None:
                return 0.0
            return rec["total_frames"] / max(1, rec["fps"])

    def stop_all_footage(self):
        """Close all active VideoWriters. Returns {visit_id: (fname, visible_secs)}."""
        results = {}
        with self._writers_lock:
            for vid, rec in list(self._active_writers.items()):
                try:
                    rec["writer"].release()
                except Exception:
                    pass
                visible_secs = rec["total_frames"] / max(1, rec["fps"])
                results[vid] = (rec["fname"], visible_secs)
            self._active_writers.clear()
        return results

    def get_qr(self):
        """Return latest decoded QR payload (string) or None."""
        with self._lock:
            return self._latest_qr

    def get_qr_state(self):
        """Return (latest_qr_payload, last_decoded_monotonic_ts)."""
        with self._lock:
            return self._latest_qr, float(self._latest_qr_t)

    def process_video_file(self, input_path, output_path, progress_cb=None):
        """Run face recognition on a saved video and write an annotated output video.

        Returns:
            dict: {"output_path": str, "mime": str}
        """
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError("Cannot open uploaded video")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
        if width <= 0 or height <= 0:
            cap.release()
            raise RuntimeError("Invalid video dimensions")

        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        out_fps = src_fps if src_fps > 0 else float(max(1, self.out_fps))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        out_w = int(width - (width % 2))
        out_h = int(height - (height % 2))
        if out_w <= 0 or out_h <= 0:
            cap.release()
            raise RuntimeError("Invalid output video dimensions")

        base_path, _ = os.path.splitext(output_path)
        writer = None
        selected_path = None
        selected_mime = None
        candidates = (
            ("mp4", "avc1", "video/mp4"),
            ("mp4", "H264", "video/mp4"),
            ("webm", "VP80", "video/webm"),
            ("webm", "VP90", "video/webm"),
            ("mp4", "mp4v", "video/mp4"),
        )
        for ext, tag, mime in candidates:
            candidate_path = f"{base_path}.{ext}"
            try:
                if os.path.exists(candidate_path):
                    os.remove(candidate_path)
            except Exception:
                pass
            w = cv2.VideoWriter(
                candidate_path,
                cv2.VideoWriter_fourcc(*tag),
                out_fps,
                (out_w, out_h),
            )
            if w.isOpened():
                writer = w
                selected_path = candidate_path
                selected_mime = mime
                break
            w.release()

        if writer is None or not selected_path or not selected_mime:
            cap.release()
            raise RuntimeError("Cannot create output video writer")

        frame_idx = 0
        video_tracks = []
        track_max_misses = 5
        bbox_smooth_alpha = 0.55

        # --- Lazy-create head detector for offline processing ---
        _local_head = self._head_detector
        _created_head = False
        if _local_head is None:
            try:
                _local_head = HeadDetector()
                _created_head = True
            except Exception:
                _local_head = None

        # --- Lazy-create action detector for offline processing ---
        _local_action = None
        _created_action = False
        if self.activity_enabled:
            _local_action = self._action_detector
            if _local_action is None:
                try:
                    _local_action = ActionDetector()
                    _created_action = True
                except Exception:
                    _local_action = None

        action_results = {}  # name -> (label, conf)
        action_detect_interval = max(1, int(out_fps * 2))  # every ~2s of video
        head_grace_frames = max(8, int(out_fps * 0.5))  # ~0.5s of video

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape[1] != out_w or frame.shape[0] != out_h:
                    frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)

                results = self._detect_and_match_faces(
                    frame,
                    min_face_size=12,
                    keep_unembedded_unknown=True,
                )
                results = self._dedupe_detections(results)

                for t in video_tracks:
                    t["matched"] = False

                results = sorted(results, key=lambda r: int(r[2]) * int(r[3]), reverse=True)
                for x, y, w, h, name, best in results:
                    det_bbox = (int(x), int(y), int(w), int(h))
                    best_i = -1
                    best_key = None
                    for i, t in enumerate(video_tracks):
                        if t.get("matched"):
                            continue
                        t_name = t.get("name", "unknown")
                        compatible = (t_name == name) or (t_name == "unknown") or (name == "unknown")
                        if not compatible:
                            continue

                        ov = iou(t.get("bbox", (0, 0, 0, 0)), det_bbox)
                        cd = self._center_dist_norm(t.get("bbox", (0, 0, 0, 0)), det_bbox)
                        if ov < 0.08 and cd > 0.95:
                            continue
                        key = (ov, -cd, 1 if t_name == name else 0)
                        if best_key is None or key > best_key:
                            best_key = key
                            best_i = i

                    if best_i >= 0:
                        t = video_tracks[best_i]
                        bx, by, bw, bh = t.get("bbox", det_bbox)
                        nx, ny, nw, nh = det_bbox
                        t["bbox"] = (
                            int(round(bx * bbox_smooth_alpha + nx * (1.0 - bbox_smooth_alpha))),
                            int(round(by * bbox_smooth_alpha + ny * (1.0 - bbox_smooth_alpha))),
                            int(round(bw * bbox_smooth_alpha + nw * (1.0 - bbox_smooth_alpha))),
                            int(round(bh * bbox_smooth_alpha + nh * (1.0 - bbox_smooth_alpha))),
                        )
                        if name != "unknown":
                            t["name"] = name
                            t["best"] = float(best)
                            t["unknown_hits"] = 0
                        else:
                            t["unknown_hits"] = int(t.get("unknown_hits", 0)) + 1
                            if t.get("name") == "unknown":
                                t["best"] = float(best)
                            elif t["unknown_hits"] >= 4:
                                t["name"] = "unknown"
                                t["best"] = float(best)
                        t["matched"] = True
                        t["misses"] = 0
                        t["last_detect_t"] = frame_idx
                        t["last_head_t"] = frame_idx  # face confirmed → head confirmed
                    else:
                        video_tracks.append(
                            {
                                "bbox": det_bbox,
                                "name": name,
                                "best": float(best),
                                "matched": True,
                                "misses": 0,
                                "unknown_hits": 0 if name != "unknown" else 1,
                                "last_detect_t": frame_idx,
                                "last_head_t": frame_idx,
                            }
                        )

                # --- Track pruning (head-detector-aware) ---
                kept_tracks = []
                for t in video_tracks:
                    if not t.get("matched"):
                        t["misses"] = int(t.get("misses", 0)) + 1
                    misses = int(t.get("misses", 0))
                    is_known = t.get("name", "unknown") != "unknown"
                    head_fresh = (frame_idx - t.get("last_head_t", 0)) <= head_grace_frames
                    max_m = track_max_misses
                    if is_known and head_fresh:
                        max_m = track_max_misses * 2  # head visible → more patient
                    if misses <= max_m:
                        kept_tracks.append(t)
                video_tracks = self._dedupe_tracks(kept_tracks, iou_thresh=0.50, center_dist_thresh=0.65)

                # --- Head detection: sustain tracks when face not visible ---
                if _local_head is not None:
                    try:
                        head_bboxes = _local_head.detect(frame)
                    except Exception:
                        head_bboxes = []
                    self._match_heads_to_tracks(video_tracks, head_bboxes, frame_idx)

                # --- Action detection: synchronous CLIP, throttled ---
                if _local_action is not None and (frame_idx % action_detect_interval == 0):
                    for t in video_tracks:
                        tname = t.get("name", "unknown")
                        if tname == "unknown":
                            continue
                        # Only when face detector recently confirmed
                        if (frame_idx - t.get("last_detect_t", 0)) > action_detect_interval:
                            continue
                        bbox = t.get("bbox", (0, 0, 0, 0))
                        try:
                            act_label, act_conf = _local_action.detect(
                                frame, bbox, person_id=tname
                            )
                            if act_label:
                                action_results[tname] = (act_label, act_conf)
                            else:
                                action_results.pop(tname, None)
                        except Exception:
                            pass

                # --- Draw annotations ---
                for t in video_tracks:
                    x, y, w, h = map(int, t.get("bbox", (0, 0, 0, 0)))
                    name = t.get("name", "unknown")
                    is_unknown = name == "unknown"
                    color = (0, 0, 255) if is_unknown else (0, 255, 0)

                    # Keep tiny CCTV detections visible in rendered output.
                    dx, dy, dw, dh = int(x), int(y), int(w), int(h)
                    min_vis = 28
                    if dw < min_vis:
                        pad = (min_vis - dw) // 2
                        dx = max(0, dx - pad)
                        dw = min(out_w - dx, max(dw, min_vis))
                    if dh < min_vis:
                        pad = (min_vis - dh) // 2
                        dy = max(0, dy - pad)
                        dh = min(out_h - dy, max(dh, min_vis))

                    cv2.rectangle(frame, (dx, dy), (dx + dw, dy + dh), color, 2)
                    if is_unknown:
                        label = "unknown"
                    else:
                        act = action_results.get(name)
                        label = f"{name} | {act[0]}" if act else name
                    cv2.putText(
                        frame,
                        label,
                        (dx, max(0, dy - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2,
                    )

                writer.write(frame)
                frame_idx += 1

                if progress_cb and (frame_idx % 5 == 0 or (total_frames and frame_idx == total_frames)):
                    try:
                        progress_cb(frame_idx, total_frames)
                    except Exception:
                        pass
        finally:
            cap.release()
            writer.release()
            # Release locally-created detectors
            if _created_head:
                _local_head = None
            if _created_action:
                _local_action = None

        if progress_cb:
            try:
                progress_cb(frame_idx, total_frames)
            except Exception:
                pass

        return {"output_path": selected_path, "mime": selected_mime}

    @staticmethod
    def _center_dist_norm(a, b):
        ax = float(a[0]) + float(a[2]) * 0.5
        ay = float(a[1]) + float(a[3]) * 0.5
        bx = float(b[0]) + float(b[2]) * 0.5
        by = float(b[1]) + float(b[3]) * 0.5
        dx = ax - bx
        dy = ay - by
        d = (dx * dx + dy * dy) ** 0.5
        scale = max(
            1.0,
            float(max(a[2], a[3])) * 0.5 + float(max(b[2], b[3])) * 0.5,
        )
        return d / scale

    def _dedupe_tracks(self, tracks, iou_thresh=0.45, center_dist_thresh=0.5):
        """Drop duplicate tracks.

        Two-pass dedup:
        1. Overlap-based — merge tracks whose bboxes overlap (IoU or center
           distance) and have compatible identities.
        2. Name-based — for each *known* name, keep only the track with the
           most recent detector confirmation (``last_detect_t``).  This kills
           ghost boxes that linger at the old position after a person moves.
        """
        if len(tracks) <= 1:
            return tracks

        def _area(b):
            return float(max(0.0, b[2]) * max(0.0, b[3]))

        def _rank(t):
            known = 0 if t.get("name") == "unknown" else 1
            return (known, float(t.get("last_seen", 0.0)), _area(t.get("bbox", (0, 0, 0, 0))))

        def _can_merge(a, b):
            a_bbox = a.get("bbox", (0, 0, 0, 0))
            b_bbox = b.get("bbox", (0, 0, 0, 0))
            ov = iou(a_bbox, b_bbox)
            if ov < iou_thresh:
                an = a.get("name", "unknown")
                bn = b.get("name", "unknown")
                cd = self._center_dist_norm(a_bbox, b_bbox)
                if an == bn and an != "unknown" and cd < center_dist_thresh:
                    return True
                if ((an == "unknown") ^ (bn == "unknown")) and cd < center_dist_thresh:
                    return True
                return False

            an = a.get("name", "unknown")
            bn = b.get("name", "unknown")
            return (an == bn) or (an == "unknown") or (bn == "unknown")

        # Pass 1: overlap-based dedup
        kept = []
        for t in sorted(tracks, key=_rank, reverse=True):
            duplicate = False
            for k in kept:
                if _can_merge(t, k):
                    duplicate = True
                    break
            if not duplicate:
                kept.append(t)

        # Pass 2: name-based dedup — one box per known person, keep latest
        seen_names = {}   # name -> index in final list
        final = []
        for t in kept:
            name = t.get("name", "unknown")
            if name == "unknown":
                final.append(t)
                continue
            prev_idx = seen_names.get(name)
            if prev_idx is None:
                seen_names[name] = len(final)
                final.append(t)
            else:
                # Keep the one with the more recent detector confirmation
                if float(t.get("last_detect_t", 0)) > float(final[prev_idx].get("last_detect_t", 0)):
                    final[prev_idx] = t
        return final

    def _dedupe_detections(self, detections, iou_thresh=0.45, center_dist_thresh=0.5):
        """Collapse near-duplicate detections from the same detector pass."""
        if len(detections) <= 1:
            return detections

        def _rank(det):
            x, y, w, h, name, best = det
            known = 1 if name != "unknown" else 0
            quality = (1.0 - float(best)) if known else 0.0
            area = int(w) * int(h)
            return (known, quality, area)

        kept = []
        for det in sorted(detections, key=_rank, reverse=True):
            x, y, w, h, name, _best = det
            bbox = (int(x), int(y), int(w), int(h))
            is_dup = False

            for kd in kept:
                kx, ky, kw, kh, kname, _kbest = kd
                kbbox = (int(kx), int(ky), int(kw), int(kh))
                compatible = (name == kname) or (name == "unknown") or (kname == "unknown")
                if not compatible:
                    continue

                ov = iou(bbox, kbbox)
                cd = self._center_dist_norm(bbox, kbbox)
                if ov > iou_thresh or cd < center_dist_thresh:
                    is_dup = True
                    break

            if not is_dup:
                kept.append(det)

        return kept

    # ---------- head detection helpers ----------

    def _run_head_detection(self, frame):
        """Run head detector on *frame* and return list of (x,y,w,h,conf).

        Returns an empty list if the head detector is not available.
        """
        if self._head_detector is None:
            return []
        try:
            return self._head_detector.detect(frame)
        except Exception:
            return []

    @staticmethod
    def _match_heads_to_tracks(tracks, head_bboxes, now):
        """Match head detections to existing tracks and set ``last_head_t``.

        A head bbox matches a track if:
        - IoU > 0.15 (heads are slightly larger than face boxes), OR
        - The centre of the head bbox falls within the track bbox.

        Only *existing* tracks are updated — head-only detections that
        don't overlap any track are ignored (we never create identity
        from a head alone).
        """
        if not tracks or not head_bboxes:
            return

        for t in tracks:
            tx, ty, tw, th = t.get("bbox", (0, 0, 0, 0))
            if tw <= 0 or th <= 0:
                continue
            for hx, hy, hw, hh, _hconf in head_bboxes:
                # Centre of head bbox
                hcx = hx + hw / 2
                hcy = hy + hh / 2
                # Check if head centre falls inside track bbox
                inside = (tx <= hcx <= tx + tw) and (ty <= hcy <= ty + th)
                if inside or iou((tx, ty, tw, th), (hx, hy, hw, hh)) > 0.15:
                    t["last_head_t"] = now
                    break  # one head match per track is enough

    # ---------- auto-capture unknowns ----------

    def _next_unknown_name(self):
        """Scan known_dir for unknown_N folders, return (next_name, existing_count)."""
        import re as _re
        existing = []
        if os.path.isdir(self.known_dir):
            for d in os.listdir(self.known_dir):
                m = _re.match(r"^unknown_(\d+)$", d)
                if m and os.path.isdir(os.path.join(self.known_dir, d)):
                    existing.append(int(m.group(1)))
        next_id = max(existing, default=0) + 1
        return f"unknown_{next_id}", len(existing)

    def _try_capture_unknown(self, track, frame, now, crop_frame=None, crop_bbox=None, pending=None):
        """
        Watch an 'unknown' track over time. After it has been continuously
        tracked for ``_unknown_capture_min_seconds`` (~5s), save a cropped
        face image and promote the track to a named unknown_N identity.

        Parameters
        ----------
        track : dict
            The currently-tracked person (bbox in tile coordinate space).
        frame : ndarray
            The tile-size frame the tracker is operating on. Used for
            "best frame" selection by face area in the tile.
        now : float
        crop_frame : ndarray | None
            Higher-resolution frame (typically the raw RTSP frame) to
            crop the saved face from. If None, falls back to ``frame``.
        crop_bbox : tuple | None
            ``(x, y, w, h)`` of the face in ``crop_frame`` coordinates.
            Required when ``crop_frame`` is supplied.

        Returns the new name if captured, else None.
        """
        if pending is None:
            pending = self._unknown_pending
        tid = id(track)

        # Only process face-detector-confirmed unknowns
        if track.get("name") != "unknown":
            # Track was recognised — discard any pending accumulation
            pending.pop(tid, None)
            return None

        # Compute face area for quality selection
        bx, by, bw, bh = track.get("bbox", (0, 0, 0, 0))
        area = bw * bh
        if area <= 0:
            return None

        entry = pending.get(tid)
        if entry is None:
            entry = {"first_t": now, "best_frame": None, "best_bbox": None, "best_area": 0, "_log_t": 0.0}
            pending[tid] = entry
            print(f"[auto-capture] new unknown track tid={tid} (need {self._unknown_capture_min_seconds:.1f}s)")
        else:
            if (now - entry.get("_log_t", 0.0)) >= 1.0:
                entry["_log_t"] = now
                print(f"[auto-capture] tid={tid} elapsed={now - entry['first_t']:.1f}s area={area}")

        if area > entry["best_area"]:
            # Prefer cropping from the higher-resolution frame when available.
            if crop_frame is not None and crop_bbox is not None:
                entry["best_frame"] = crop_frame.copy()
                entry["best_bbox"] = tuple(int(v) for v in crop_bbox)
            else:
                entry["best_frame"] = frame.copy()
                entry["best_bbox"] = (bx, by, bw, bh)
            entry["best_area"] = area

        if (now - entry["first_t"]) < self._unknown_capture_min_seconds:
            return None

        if entry["best_frame"] is None:
            print(f"[auto-capture] tid={tid} hit time threshold but no best_frame — skipping")
            return None

        # --- Ready to capture ---
        next_name, existing_count = self._next_unknown_name()
        if existing_count >= self._unknown_max_auto:
            print(f"[auto-capture] Max {self._unknown_max_auto} auto-captured unknowns reached, skipping")
            pending.pop(tid, None)
            return None

        # Crop face with 2x padding from best frame
        bf = entry["best_frame"]
        fx, fy, fw, fh = entry["best_bbox"]
        H, W = bf.shape[:2]
        pad_w, pad_h = fw, fh  # 1x extra on each side = 2x total
        cx1 = max(0, fx - pad_w)
        cy1 = max(0, fy - pad_h)
        cx2 = min(W, fx + fw + pad_w)
        cy2 = min(H, fy + fh + pad_h)
        crop = bf[cy1:cy2, cx1:cx2]

        if crop.size == 0:
            pending.pop(tid, None)
            return None

        # Save to faces/unknown_N/1.jpg
        person_dir = os.path.join(self.known_dir, next_name)
        os.makedirs(person_dir, exist_ok=True)
        save_path = os.path.join(person_dir, "1.jpg")
        cv2.imwrite(save_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"[auto-capture] Saved {next_name} -> {save_path}  (face area={entry['best_area']})")

        # Promote track
        track["name"] = next_name
        track["unknown_hits"] = 0

        # Reload embeddings in background (expensive)
        def _bg_reload():
            try:
                self.reload_faces()
                print(f"[auto-capture] reload_faces() done, {len(self.known_embeddings)} people")
            except Exception as e:
                print(f"[auto-capture] reload_faces() error: {e}")
        threading.Thread(target=_bg_reload, daemon=True).start()

        # Cleanup
        pending.pop(tid, None)
        return next_name

    def _try_capture_more_for_known(self, track, crop_frame, crop_bbox, now, camera_source=None):
        """Save additional sample images for an already-recognised person
        (auto-captured ``unknown_N`` or human-named), up to
        ``self._unknown_max_images_per_person``.

        Per-camera cooldown: saving on the same camera is rate-limited
        by ``self._extra_capture_per_cam_cooldown``; a different camera
        sees a different viewpoint and is allowed immediately.

        Returns True if an image was saved, False otherwise.
        """
        name = str(track.get("name", ""))
        if not name or name == "unknown":
            return False

        cam_key = str(camera_source) if camera_source is not None else ""
        with self._last_extra_capture_lock:
            last_t = float(self._last_extra_capture_at.get((name, cam_key), 0.0))
        if (now - last_t) < self._extra_capture_per_cam_cooldown:
            return False

        person_dir = os.path.join(self.known_dir, name)
        if not os.path.isdir(person_dir):
            return False

        existing = [
            f for f in os.listdir(person_dir)
            if f.lower().endswith(self.allowed_exts)
        ]
        n = len(existing)
        if n >= self._unknown_max_images_per_person:
            return False

        bx, by, bw, bh = (int(v) for v in crop_bbox)
        if bw <= 0 or bh <= 0:
            return False
        H, W = crop_frame.shape[:2]
        pad_w, pad_h = bw, bh
        cx1 = max(0, bx - pad_w)
        cy1 = max(0, by - pad_h)
        cx2 = min(W, bx + bw + pad_w)
        cy2 = min(H, by + bh + pad_h)
        crop = crop_frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return False

        # Verify the new crop actually matches this person's existing
        # template before saving. Without this check, a track that was
        # promoted to unknown_N can drift onto a different person (the
        # detector treats any "unknown" detection as compatible with an
        # unknown_N track) and contaminate the folder with someone else.
        template = next(
            (emb for nm, emb in self.known_embeddings if nm == name),
            None,
        )
        if template is None:
            # reload_faces() hasn't built the template yet — skip rather
            # than save blind. We'll get another chance on the next interval.
            return False
        if template is not None:
            try:
                tight = crop_frame[by:by + bh, bx:bx + bw]
                if tight.size == 0:
                    return False
                with self._inference_lock:
                    faces = DeepFace.extract_faces(
                        img_path=tight,
                        detector_backend=self.detector,
                        enforce_detection=False,
                        align=True,
                    )
                faces = [
                    f for f in (faces or [])
                    if float(f.get("confidence", 0.0)) >= 0.90
                    and f.get("face") is not None
                ]
                if not faces:
                    return False
                with self._inference_lock:
                    rep = DeepFace.represent(
                        img_path=faces[0]["face"],
                        model_name=self.model,
                        detector_backend="skip",
                        enforce_detection=False,
                    )
                if not rep:
                    return False
                emb_new = l2norm(rep[0]["embedding"])
                d = cosine_distance(emb_new, template)
                if d > self.threshold:
                    return False
            except Exception:
                return False

        # Pick the next free integer filename. We don't assume contiguous
        # numbering — the user might have manually deleted some.
        used = set()
        for f in existing:
            stem = os.path.splitext(f)[0]
            if stem.isdigit():
                used.add(int(stem))
        idx = 1
        while idx in used:
            idx += 1
        save_path = os.path.join(person_dir, f"{idx}.jpg")
        try:
            cv2.imwrite(save_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        except Exception:
            return False
        with self._last_extra_capture_lock:
            self._last_extra_capture_at[(name, cam_key)] = now

        # Reload embeddings in the background so the new sample is used
        # the next time this person enters the frame. Cheap relative to
        # the once-per-second cap.
        def _bg_reload():
            try:
                self.reload_faces()
            except Exception:
                pass
        threading.Thread(target=_bg_reload, daemon=True).start()
        return True

    def _cleanup_unknown_pending(self, live_track_ids, pending=None):
        """Remove pending entries for tracks that no longer exist."""
        if pending is None:
            pending = self._unknown_pending
        stale = [tid for tid in pending if tid not in live_track_ids]
        for tid in stale:
            del pending[tid]

    def reload_faces(self):
        """Rebuild known embeddings from faces/{name}/*"""
        known = []
        if not os.path.isdir(self.known_dir):
            os.makedirs(self.known_dir, exist_ok=True)

        for person in os.listdir(self.known_dir):
            person_dir = os.path.join(self.known_dir, person)
            if not os.path.isdir(person_dir):
                continue

            embs = []
            for file in os.listdir(person_dir):
                if not file.lower().endswith(self.allowed_exts):
                    continue
                path = os.path.join(person_dir, file)
                try:
                    faces = DeepFace.extract_faces(
                        img_path=path,
                        detector_backend=self.detector,
                        enforce_detection=False,
                        align=True,
                    )
                    # Filter out the "no face found" placeholder DeepFace
                    # returns when enforce_detection=False (confidence==0).
                    faces = [
                        f for f in (faces or [])
                        if float(f.get("confidence", 0.0)) >= 0.90
                        and f.get("face") is not None
                    ]
                    if not faces:
                        continue
                    face_img = faces[0]["face"]
                    rep = DeepFace.represent(
                        img_path=face_img,
                        model_name=self.model,
                        detector_backend="skip",
                        enforce_detection=False,
                    )
                    if rep and "embedding" in rep[0]:
                        embs.append(l2norm(rep[0]["embedding"]))
                except Exception as e:
                    # Don't spam — bad crops are expected with auto-capture.
                    import logging as _logging
                    _logging.getLogger(__name__).debug("reload_faces: skip %s (%s)", path, e)
                    continue

            if embs:
                template = l2norm(np.mean(np.asarray(embs), axis=0))
                known.append((person, template))

        self.known_embeddings = known

    def _detect_and_match_faces(self, frame_full, min_face_size=30, keep_unembedded_unknown=False):
        H, W = frame_full.shape[:2]
        results = []

        try:
            used_scale = self.detect_scale
            small = cv2.resize(frame_full, None, fx=used_scale, fy=used_scale)

            with self._inference_lock:
                faces = DeepFace.extract_faces(
                    img_path=small,
                    detector_backend=self.detector,
                    enforce_detection=False,
                    align=True,
                )

                if not faces:
                    faces = DeepFace.extract_faces(
                        img_path=frame_full,
                        detector_backend=self.detector,
                        enforce_detection=False,
                        align=True,
                    )
                    used_scale = 1.0

            inv = 1.0 / used_scale
        except Exception as _ext_err:
            import logging as _logging
            _logging.getLogger(__name__).warning("extract_faces failed: %s", _ext_err)
            faces = []
            inv = 1.0


        for face in faces:
            fa = face.get("facial_area", {})
            x = int(fa.get("x", 0) * inv)
            y = int(fa.get("y", 0) * inv)
            w = int(fa.get("w", 0) * inv)
            h = int(fa.get("h", 0) * inv)

            x = max(0, x)
            y = max(0, y)
            w = max(1, min(w, W - x))
            h = max(1, min(h, H - y))

            if w < int(min_face_size) or h < int(min_face_size):
                continue
            if w > 0.70 * W or h > 0.70 * H:
                continue
            ar = w / float(h)
            if ar < 0.6 or ar > 1.7:
                continue
            conf = face.get("confidence", None)
            if conf is not None and conf < 0.90:
                continue

            face_img = face.get("face")
            if face_img is None or getattr(face_img, "size", 0) == 0:
                continue

            try:
                with self._inference_lock:
                    rep_live = DeepFace.represent(
                        img_path=face_img,
                        model_name=self.model,
                        detector_backend="skip",
                        enforce_detection=False,
                    )
                if not rep_live:
                    if keep_unembedded_unknown:
                        results.append((x, y, w, h, "unknown", 1.0))
                    continue
                emb_live = l2norm(rep_live[0]["embedding"])
            except Exception as _rep_err:
                import logging as _logging
                _logging.getLogger(__name__).warning("represent failed: %s", _rep_err)
                if keep_unembedded_unknown:
                    results.append((x, y, w, h, "unknown", 1.0))
                continue

            name = "unknown"
            best = 1.0
            for known_name, emb_known in self.known_embeddings:
                d = cosine_distance(emb_live, emb_known)
                if d < best:
                    best = d
                    name = known_name
            if best > self.threshold:
                name = "unknown"

            results.append((x, y, w, h, name, best))

        return results

    # ---------- internal ----------
    def _push_latest(self, q, item):
        try:
            q.put_nowait(item)
        except Full:
            try:
                q.get_nowait()
            except Empty:
                pass
            q.put_nowait(item)

    def _detector_worker(self):
        while not self.stop_evt.is_set():
            try:
                det_input = self.in_q.get(timeout=0.1)
            except Empty:
                continue

            if (
                isinstance(det_input, tuple)
                and len(det_input) == 2
            ):
                frame_t, frame_full = det_input
            else:
                frame_t = time.monotonic()
                frame_full = det_input

            results = self._detect_and_match_faces(frame_full)
            head_bboxes = self._run_head_detection(frame_full)

            # publish latest (face results + head bboxes)
            while True:
                try:
                    self.out_q.get_nowait()
                except Empty:
                    break
            self.out_q.put((float(frame_t), results, head_bboxes))

    def _is_grid_mode(self):
        return isinstance(self.cam_index, str) and self.cam_index.startswith("grid:")

    @staticmethod
    def _parse_grid_sources(cam_index):
        """Parse ``cam_index`` string into an ordered list of sources.

        Supports empty slots encoded as empty tokens between commas::

            "grid:0,,2,4,1,"  ->  [0, None, 2, 4, 1, None]

        ``None`` entries represent deliberately empty grid slots.
        """
        if not (isinstance(cam_index, str) and cam_index.startswith("grid:")):
            return []
        raw = cam_index.split(":", 1)[1].strip()
        if not raw:
            return []

        out = []
        for token in raw.split(","):
            t = token.strip()
            if not t:
                out.append(None)  # empty slot
            elif t.isdigit():
                out.append(int(t))
            else:
                out.append(t)
        return out

    # ---------- grid config persistence ----------

    def set_grid_layout(self, rows, cols):
        """Validate and set the grid layout (rows, cols)."""
        rows, cols = int(rows), int(cols)
        if (rows, cols) not in AVAILABLE_LAYOUTS:
            raise ValueError(
                f"Unsupported grid layout ({rows}, {cols}). "
                f"Supported: {AVAILABLE_LAYOUTS}"
            )
        self._grid_layout = (rows, cols)

    @staticmethod
    def _normalize_slot(val):
        """Normalize a slot value to ``{"source": str, "name": str} | None``.

        Handles both old format (plain string) and new format (dict with
        ``source`` and ``name`` keys).
        """
        if val is None:
            return None
        if isinstance(val, dict):
            src = val.get("source")
            if src is None or str(src).strip() == "":
                return None
            return {"source": str(src).strip(), "name": str(val.get("name") or f"Camera {src}")}
        # Old format: plain string
        val = str(val).strip()
        if not val:
            return None
        return {"source": val, "name": f"Camera {val}"}

    @staticmethod
    def save_grid_config(layout, slots, path=None):
        """Persist grid configuration to JSON.

        Parameters
        ----------
        layout : tuple[int, int]
            (rows, cols)
        slots : dict[str, dict | str | None]
            Mapping of slot index (as str) to ``{"source": str, "name": str}``
            or a plain source string (auto-wrapped) or None.
        path : str | None
            File path; defaults to ``GRID_CONFIG_PATH``.
        """
        path = path or GRID_CONFIG_PATH
        normalized = {}
        for k, v in slots.items():
            normalized[str(k)] = FaceEngine._normalize_slot(v)
        data = {
            "layout": list(layout),
            "slots": normalized,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load_grid_config(path=None):
        """Load grid configuration from JSON.

        Handles both old format (slot values are plain strings) and new format
        (slot values are ``{"source": ..., "name": ...}`` dicts).

        Returns
        -------
        dict | None
            ``{"layout": (rows, cols), "slots": {str: dict|None}, "cam_index": str}``
            or ``None`` if the file does not exist or is invalid.
        """
        path = path or GRID_CONFIG_PATH
        if not os.path.isfile(path):
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            layout = tuple(data.get("layout", [2, 2]))
            if len(layout) != 2:
                return None
            layout = (int(layout[0]), int(layout[1]))
            raw_slots = data.get("slots", {})

            # Normalize all slots to new format
            slots = {}
            for k, v in raw_slots.items():
                slots[str(k)] = FaceEngine._normalize_slot(v)

            # Build cam_index string from slots
            cam_index = FaceEngine.build_grid_cam_index(slots)
            return {"layout": layout, "slots": slots, "cam_index": cam_index}
        except Exception:
            return None

    @staticmethod
    def build_grid_cam_index(slots):
        """Build a ``grid:...`` cam_index string from a slots dict.

        Parameters
        ----------
        slots : dict[str, dict | str | None]
            Slot index -> ``{"source": ..., "name": ...}`` or plain source string or None.

        Returns
        -------
        str
            e.g. ``"grid:0,,2,4,1,"``
        """
        if not slots:
            return "grid:"
        max_slot = max((int(k) for k in slots), default=-1)
        parts = []
        for i in range(max_slot + 1):
            val = slots.get(str(i))
            if val is None:
                parts.append("")
            elif isinstance(val, dict):
                src = val.get("source", "")
                parts.append(str(src) if src else "")
            else:
                parts.append(str(val) if val else "")
        return "grid:" + ",".join(parts)

    @staticmethod
    def get_slot_locations(slots):
        """Extract a mapping of camera_source -> location name from slots.

        Returns
        -------
        dict[str, str]
            e.g. ``{"0": "Front Entrance", "2": "Lobby"}``
        """
        locations = {}
        if not slots:
            return locations
        for v in slots.values():
            if v is None:
                continue
            if isinstance(v, dict):
                src = v.get("source")
                name = v.get("name")
                if src:
                    locations[str(src)] = str(name or f"Camera {src}")
        return locations

    def _start_grid_mode(self):
        sources = self._parse_grid_sources(self.cam_index)
        if not sources:
            raise RuntimeError("No camera sources configured for grid mode")

        rows, cols = self._grid_layout
        # Fixed 16:9 tile aspect ratio regardless of grid layout
        cell_w = max(1, int(self.width // cols))
        cell_h = max(1, int(self.height // rows))
        target_ratio = self.width / max(1, self.height)
        if cell_w / max(1, cell_h) > target_ratio:
            tile_h = cell_h
            tile_w = max(1, int(tile_h * target_ratio))
        else:
            tile_w = cell_w
            tile_h = max(1, int(tile_w / target_ratio))
        workers = {}
        # Ordered list with None entries preserved so each slot index
        # maps directly to a grid position. Workers are created for ALL
        # configured sources (not just those visible in the current grid
        # layout) so face recognition and visit tracking run on every
        # camera regardless of what the viewer is looking at.
        ordered_sources = []
        self._grid_stop_evt.clear()
        self._grid_qr_rr_idx = 0

        for source in sources:
            # None means an intentionally empty slot — no worker needed
            if source is None:
                ordered_sources.append(None)
                continue

            # Skip duplicate sources — a camera can only be opened once
            src_key = str(source)
            if src_key in workers:
                ordered_sources.append(None)
                continue

            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                try:
                    cap.release()
                except Exception:
                    pass
                ordered_sources.append(None)  # treat as empty if can't open
                continue

            # IMPORTANT: do NOT call cap.set(CAP_PROP_FRAME_WIDTH/HEIGHT).
            # That would downscale the camera feed at the source on backends
            # that honor it, starving the AI detectors with a tile-sized
            # frame. We want the camera's native resolution for detection
            # and only resize for the tile renderer (done in capture loop).
            if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass

            # Phase-stagger per worker so N grid workers don't all hit
            # the inference lock at the same moment. Spreads the load
            # across detect_every and dramatically reduces lock contention
            # on large grids (10+ cams).
            _worker_idx = len(workers)
            _detect_period = max(0.25, float(self.detect_every))
            _phase_offset = (_detect_period * _worker_idx / max(1, len(sources))) - _detect_period

            worker = {
                "source": source,
                "cap": cap,
                "lock": threading.Lock(),
                "stop_evt": threading.Event(),
                "frame_q": Queue(maxsize=1),
                "latest_frame": None,
                "latest_raw_frame": None,
                "latest_frame_t": 0.0,
                "last_good_frame_t": 0.0,
                "tracks": [],
                "latest_tracks": [],
                "last_det_t": time.monotonic() + _phase_offset,
                "track_max_misses": 6,
                "bbox_smooth_alpha": 0.55,
                "max_unknown_tracks": 3,
                "capture_thread": None,
                "detect_thread": None,
                # Per-worker pending dict for auto-capture so workers don't
                # wipe each other's state via the engine-level cleanup.
                "unknown_pending": {},
            }
            worker["capture_thread"] = threading.Thread(
                target=self._grid_capture_loop,
                args=(worker, tile_w, tile_h),
                daemon=True,
            )
            worker["detect_thread"] = threading.Thread(
                target=self._grid_detect_loop,
                args=(worker,),
                daemon=True,
            )
            workers[src_key] = worker
            ordered_sources.append(src_key)

        if not workers:
            import logging
            logging.getLogger(__name__).warning(
                "Grid mode: no cameras opened — all tiles will show 'No Signal'. "
                "Check that at least one configured camera is connected."
            )

        self._grid_workers = workers
        self._grid_sources = ordered_sources

        for source in self._grid_sources:
            if source is None:
                continue
            w = self._grid_workers.get(source)
            if w is not None:
                w["capture_thread"].start()
                w["detect_thread"].start()

        self._grid_render_t = threading.Thread(target=self._grid_render_loop, daemon=True)
        self._grid_render_t.start()

    def _cleanup_grid_mode(self):
        # 1) Signal every worker to exit.
        self._grid_stop_evt.set()
        for w in self._grid_workers.values():
            try:
                w["stop_evt"].set()
            except Exception:
                pass

        import logging
        _log = logging.getLogger(__name__)

        # 2) Join workers BEFORE releasing their cv2.VideoCapture handles.
        # On RTSP, cap.read() can block for up to ~stimeout (5s) since the
        # stop flag is only checked between reads — releasing the cap while
        # the worker is still inside read() would be a use-after-free and
        # segfault the process. Use a join timeout > stimeout so RTSP
        # workers actually have a chance to exit.
        joined = {}
        for src, w in self._grid_workers.items():
            ct = w.get("capture_thread")
            ok = True
            if ct is not None:
                ct.join(timeout=8.0)
                if ct.is_alive():
                    _log.warning("_cleanup_grid: capture_thread [%s] zombie — leaking cap to avoid crash", w.get("source", "?"))
                    ok = False
            dt = w.get("detect_thread")
            if dt is not None:
                dt.join(timeout=4.0)
                if dt.is_alive():
                    _log.warning("_cleanup_grid: detect_thread [%s] zombie", w.get("source", "?"))
            joined[src] = ok

        # 3) Release caps only for workers that actually stopped. A zombie
        # capture thread still holds a reference to its cap; releasing it
        # would be a use-after-free.
        for src, w in self._grid_workers.items():
            if not joined.get(src, False):
                continue
            cap = w.get("cap")
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

        if self._grid_render_t is not None:
            self._grid_render_t.join(timeout=3.0)
            if self._grid_render_t.is_alive():
                _log.warning("_cleanup_grid: render_thread zombie")
            self._grid_render_t = None

        self._grid_workers = {}
        self._grid_sources = []
        self._grid_qr_rr_idx = 0
        self._grid_stop_evt.clear()

    def _draw_no_signal_tile(self, tile, source):
        h, w = tile.shape[:2]
        cv2.rectangle(tile, (0, 0), (w - 1, h - 1), (40, 40, 120), 2)
        cv2.putText(
            tile,
            f"No Signal [{source}]",
            (12, max(24, h // 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    def _draw_empty_tile(self, tile):
        h, w = tile.shape[:2]
        cv2.rectangle(tile, (0, 0), (w - 1, h - 1), (60, 60, 60), 1)
        cv2.putText(
            tile,
            "Empty",
            (12, max(24, h // 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (150, 150, 150),
            2,
        )

    def _grid_capture_loop(self, worker, tile_w, tile_h):
        cap = worker["cap"]
        while not self.stop_evt.is_set() and not self._grid_stop_evt.is_set() and not worker["stop_evt"].is_set():
            ok, frame = cap.read()
            now = time.monotonic()
            if not ok or frame is None:
                with worker["lock"]:
                    if (now - float(worker.get("last_good_frame_t", 0.0))) > 1.0:
                        worker["latest_frame"] = None
                time.sleep(0.02)
                continue

            raw_frame = frame  # original camera resolution — used for footage
            try:
                frame = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
            except Exception:
                time.sleep(0.01)
                continue

            with worker["lock"]:
                worker["latest_frame"] = frame
                worker["latest_raw_frame"] = raw_frame
                worker["latest_frame_t"] = now
                worker["last_good_frame_t"] = now

            # Push BOTH the raw camera frame AND the tile-sized resize.
            # Detection runs on raw for accuracy on small/distant faces;
            # the tracker is initialised in tile space so render-loop
            # tracker.update(tile) calls remain cheap and consistent.
            self._push_latest(
                worker["frame_q"],
                (now, raw_frame.copy(), frame.copy()),
            )

    def _grid_detect_loop(self, worker):
        UNKNOWN_TO_FORGET = 3
        detect_period = max(0.25, float(self.detect_every))
        while not self.stop_evt.is_set() and not self._grid_stop_evt.is_set() and not worker["stop_evt"].is_set():
            try:
                _frame_t, raw_frame, tile_frame = worker["frame_q"].get(timeout=0.1)
            except Empty:
                continue
            except (TypeError, ValueError):
                # Defensive: queue payload shape mismatch (e.g. legacy 2-tuple).
                continue

            now = time.monotonic()
            with worker["lock"]:
                last_det_t = float(worker.get("last_det_t", 0.0))
            if (now - last_det_t) < detect_period:
                continue

            # Run face detection on the FULL-resolution camera frame so
            # small faces are still found, then scale results into the
            # tile coordinate system the tracker + renderer expect.
            try:
                dets_raw = self._detect_and_match_faces(
                    raw_frame,
                    min_face_size=20,
                    keep_unembedded_unknown=True,
                )
                dets_raw = self._dedupe_detections(dets_raw)
            except Exception as _det_err:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "grid detect failed for %s: %s", worker.get("source", "?"), _det_err,
                )
                dets_raw = []

            tile_h_local, tile_w_local = tile_frame.shape[:2]
            raw_h_local, raw_w_local = raw_frame.shape[:2]
            sx = tile_w_local / max(1, raw_w_local)
            sy = tile_h_local / max(1, raw_h_local)

            # Apply noisy-unknown filtering in RAW coordinates before
            # scaling. Doing this in tile space (post-scale) was rejecting
            # legitimate ~80px raw faces that shrank below 20px in the
            # tile, killing auto-capture and action detection downstream.
            raw_min_face = 20
            raw_margin_x = max(3, int(0.01 * raw_w_local))
            raw_margin_y = max(3, int(0.01 * raw_h_local))

            dets = []
            for x, y, w, h, name, best in dets_raw:
                if name == "unknown":
                    if int(w) < raw_min_face or int(h) < raw_min_face:
                        continue
                    if (x <= raw_margin_x) or (y <= raw_margin_y) \
                       or ((x + w) >= (raw_w_local - raw_margin_x)) \
                       or ((y + h) >= (raw_h_local - raw_margin_y)):
                        continue
                tx = int(round(x * sx))
                ty = int(round(y * sy))
                tw = max(1, int(round(w * sx)))
                th = max(1, int(round(h * sy)))
                dets.append((tx, ty, tw, th, name, best))

            # --- Grab tracks snapshot under lock (fast) ---
            with worker["lock"]:
                tracks = list(worker.get("tracks", []))
                bbox_smooth_alpha = float(worker.get("bbox_smooth_alpha", 0.55))
                track_max_misses = int(worker.get("track_max_misses", 6))
                max_unknown_tracks = int(worker.get("max_unknown_tracks", 3))

            # --- Head detection on the raw frame too (better for small
            # heads); scale resulting boxes back to tile space. The
            # downstream matcher expects 5-tuples (x, y, w, h, conf).
            head_bboxes_raw = self._run_head_detection(raw_frame)
            head_bboxes = []
            for hbb in head_bboxes_raw:
                if not hbb or len(hbb) < 4:
                    continue
                hx, hy, hw, hh = hbb[:4]
                hconf = float(hbb[4]) if len(hbb) >= 5 else 1.0
                head_bboxes.append((
                    int(round(hx * sx)),
                    int(round(hy * sy)),
                    max(1, int(round(hw * sx))),
                    max(1, int(round(hh * sy))),
                    hconf,
                ))

            # `frame` is what the rest of the loop (tracker init, auto-
            # capture) operates on. Keep it tile-sized so tracker.update
            # calls in the render loop remain consistent.
            frame = tile_frame

            # --- Track matching + update (no lock needed — local list) ---
            matched_track_idx = set()
            Hf, Wf = frame.shape[:2]

            # Match larger detections first for more stable assignment.
            # The noisy-unknown filter was already applied in RAW space
            # above (against the camera-native dimensions); doing it
            # again in tile space would over-filter.
            dets = sorted(dets, key=lambda r: int(r[2]) * int(r[3]), reverse=True)
            for x, y, w, h, name, best in dets:
                det_bbox = (int(x), int(y), int(w), int(h))
                best_i = -1
                best_key = None
                for i, t in enumerate(tracks):
                    if i in matched_track_idx:
                        continue
                    t_name = t.get("name", "unknown")
                    compatible = (t_name == name) or (t_name == "unknown") or (name == "unknown")
                    if not compatible:
                        continue

                    t_bbox = t.get("bbox", (0, 0, 0, 0))
                    center_dist = self._center_dist_norm(t_bbox, det_bbox)
                    score = iou(t_bbox, det_bbox)
                    motion_match = (
                        (t_name == name and name != "unknown" and center_dist < 0.75)
                        or (((t_name == "unknown") ^ (name == "unknown")) and center_dist < 0.50)
                        or (t_name == "unknown" and name == "unknown" and center_dist < 0.40)
                    )
                    if score <= 0.3 and not motion_match:
                        continue

                    cand_key = (score, 1 if t_name != "unknown" else 0, -center_dist)
                    if (best_key is None) or (cand_key > best_key):
                        best_key = cand_key
                        best_i = i

                if best_i >= 0:
                    t = tracks[best_i]
                    matched_track_idx.add(best_i)
                    bx, by, bw, bh = t.get("bbox", det_bbox)
                    nx, ny, nw, nh = det_bbox
                    smooth_bbox = (
                        int(round(bx * bbox_smooth_alpha + nx * (1.0 - bbox_smooth_alpha))),
                        int(round(by * bbox_smooth_alpha + ny * (1.0 - bbox_smooth_alpha))),
                        int(round(bw * bbox_smooth_alpha + nw * (1.0 - bbox_smooth_alpha))),
                        int(round(bh * bbox_smooth_alpha + nh * (1.0 - bbox_smooth_alpha))),
                    )
                    t["bbox"] = smooth_bbox
                    try:
                        tr = create_tracker(self.tracker_type)
                        tr.init(frame, smooth_bbox)
                        t["tracker"] = tr
                    except Exception:
                        pass
                    t["updated"] = True
                    t["last_seen"] = now
                    t["last_detect_t"] = now
                    t["last_head_t"] = now   # face confirmed ⇒ head confirmed too
                    t["misses"] = 0

                    if name == "unknown":
                        if t.get("name") != "unknown":
                            t["unknown_hits"] = int(t.get("unknown_hits", 0)) + 1
                            if t["unknown_hits"] >= UNKNOWN_TO_FORGET:
                                t["name"] = "unknown"
                                t["best"] = float(best)
                        else:
                            t["best"] = float(best)
                    else:
                        t["name"] = name
                        t["best"] = float(best)
                        t["unknown_hits"] = 0
                else:
                    near_existing = False
                    for t in tracks:
                        t_name = t.get("name", "unknown")
                        compatible = (t_name == name) or (t_name == "unknown") or (name == "unknown")
                        if not compatible:
                            continue
                        ov = iou(t.get("bbox", (0, 0, 0, 0)), det_bbox)
                        cd = self._center_dist_norm(t.get("bbox", (0, 0, 0, 0)), det_bbox)
                        if ov > 0.45 or cd < 0.32:
                            near_existing = True
                            break
                    if near_existing:
                        continue
                    if name == "unknown":
                        unknown_count = sum(1 for t in tracks if t.get("name") == "unknown")
                        if unknown_count >= max_unknown_tracks:
                            continue

                    try:
                        tr = create_tracker(self.tracker_type)
                        tr.init(frame, det_bbox)
                    except Exception:
                        continue
                    tracks.append(
                        {
                            "tracker": tr,
                            "name": name,
                            "best": float(best),
                            "bbox": det_bbox,
                            "updated": True,
                            "last_seen": now,
                            "last_detect_t": now,
                            "last_head_t": now,
                            "misses": 0,
                            "unknown_hits": 0 if name != "unknown" else 1,
                        }
                    )

            for i, t in enumerate(tracks):
                if i not in matched_track_idx:
                    t["misses"] = int(t.get("misses", 0)) + 1

            tracks = [t for t in tracks if int(t.get("misses", 0)) <= track_max_misses]
            tracks = self._dedupe_tracks(tracks)

            # --- Head matching (cheap — already ran inference above) ---
            self._match_heads_to_tracks(tracks, head_bboxes, now)

            # --- Auto-capture unknowns ---
            # Crop from the raw RTSP frame for highest enrollment quality.
            # We have to convert the tile-space track bbox back to raw
            # coordinates using the inverse of (sx, sy).
            if self.auto_capture_enabled:
                inv_sx = raw_w_local / max(1, tile_w_local)
                inv_sy = raw_h_local / max(1, tile_h_local)
                for t in tracks:
                    if not t.get("updated"):
                        continue
                    bx, by, bw, bh = t.get("bbox", (0, 0, 0, 0))
                    raw_bbox = (
                        int(round(bx * inv_sx)),
                        int(round(by * inv_sy)),
                        max(1, int(round(bw * inv_sx))),
                        max(1, int(round(bh * inv_sy))),
                    )
                    if t.get("name") == "unknown":
                        # First-time enrollment of a new unknown person.
                        captured = self._try_capture_unknown(
                            t, frame, now,
                            crop_frame=raw_frame, crop_bbox=raw_bbox,
                            pending=worker["unknown_pending"],
                        )
                        if captured:
                            print(f"[grid-detect] auto-captured {captured} on cam {worker['source']}")
                    else:
                        # Recognised (auto-captured unknown_N or human-named):
                        # top up samples, throttled per (person, camera).
                        self._try_capture_more_for_known(
                            t, raw_frame, raw_bbox, now,
                            camera_source=worker.get("source"),
                        )
                self._cleanup_unknown_pending(
                    {id(t) for t in tracks}, pending=worker["unknown_pending"]
                )

            # --- Action detection (enqueue to async thread, throttled) ---
            # Only enqueue when face detector recently confirmed the person;
            # head-detector-only or tracker-only boxes (e.g. back of head) are skipped.
            if self.activity_enabled:
                _n_workers_act = max(1, len(self._grid_workers or {}))
                face_fresh_secs = max(1.0, self.detect_every * 2.0 * _n_workers_act)
                # Send the raw camera frame + raw-space bbox to CLIP so
                # body cues (clothing, posture) aren't degraded by the
                # downscale to tile size.
                inv_sx = raw_w_local / max(1, tile_w_local)
                inv_sy = raw_h_local / max(1, tile_h_local)
                for t in tracks:
                    name = t.get("name", "unknown")
                    if name == "unknown":
                        continue
                    if (now - float(t.get("last_detect_t", 0.0))) > face_fresh_secs:
                        continue
                    last_act_t = float(t.get("_last_activity_t", 0.0))
                    if (now - last_act_t) < self.activity_detect_every:
                        continue
                    bx, by, bw, bh = t.get("bbox", (0, 0, 0, 0))
                    raw_bbox = (
                        int(round(bx * inv_sx)),
                        int(round(by * inv_sy)),
                        max(1, int(round(bw * inv_sx))),
                        max(1, int(round(bh * inv_sy))),
                    )
                    try:
                        self._activity_q.put_nowait((raw_frame.copy(), raw_bbox, name))
                    except Full:
                        pass
                    t["_last_activity_t"] = now

            # --- Write results back under lock (fast) ---
            with worker["lock"]:
                worker["tracks"] = tracks
                worker["latest_tracks"] = [
                    {
                        "name": t.get("name", "unknown"),
                        "best": float(t.get("best", 1.0)),
                        "bbox": [
                            int(t.get("bbox", (0, 0, 0, 0))[0]),
                            int(t.get("bbox", (0, 0, 0, 0))[1]),
                            int(t.get("bbox", (0, 0, 0, 0))[2]),
                            int(t.get("bbox", (0, 0, 0, 0))[3]),
                        ],
                        "activity": self.get_activity(t.get("name", "unknown"))[0],
                        "activity_conf": self.get_activity(t.get("name", "unknown"))[1],
                    }
                    for t in tracks
                ]
                worker["last_det_t"] = now

    def _grid_render_loop(self):
        rows, cols = self._grid_layout
        # Fixed 16:9 tile aspect ratio regardless of grid layout
        cell_w = max(1, int(self.width // cols))
        cell_h = max(1, int(self.height // rows))
        target_ratio = self.width / max(1, self.height)
        if cell_w / max(1, cell_h) > target_ratio:
            tile_h = cell_h
            tile_w = max(1, int(tile_h * target_ratio))
        else:
            tile_w = cell_w
            tile_h = max(1, int(tile_w / target_ratio))
        grid_w = cell_w * cols
        grid_h = cell_h * rows
        last_encode_t = 0.0
        last_single_encode_t = 0.0
        stale_secs = 1.6
        unknown_stale_secs = max(0.8, self.detect_every * 1.0)

        canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)  # reused each frame
        last_snap_t = 0.0  # throttle snapshot JPEG encoding

        # Build a list of every source we have a worker for. The visible
        # composite is only the first rows*cols of the *configured*
        # grid_sources, but we still want to run track upkeep + visit
        # detection for every running worker.
        analysis_sources = list(self._grid_sources)
        for src in self._grid_workers.keys():
            if src not in analysis_sources:
                analysis_sources.append(src)

        # Reconfirm windows scale with the number of cameras: detection is
        # serialized through _inference_lock, so per-camera detect cadence
        # is roughly num_workers * detect_every. Tracks must outlive that
        # window or every track gets killed before reconfirmation.
        n_workers = max(1, len([s for s in analysis_sources if s is not None]))
        unknown_reconfirm_secs = max(3.0, self.detect_every * 5.0 * n_workers)
        known_reconfirm_secs = max(5.0, self.detect_every * 7.0 * n_workers)

        while not self.stop_evt.is_set() and not self._grid_stop_evt.is_set():
            now = time.monotonic()
            canvas[:] = 0  # clear reused buffer instead of allocating
            aggregated_tracks = []

            # Snapshot the page offset once per frame so the visible
            # window doesn't change mid-render if the user clicks the
            # arrow concurrently.
            visible_start = max(0, int(self.viewer_grid_offset))
            visible_end = visible_start + (rows * cols)

            # Iterate over every analysis source (not just the visible grid
            # slots). The slot index decides whether we paint into the
            # composite; everything else (tracker update, snapshots, visit
            # bookkeeping) runs for all sources.
            for slot, source in enumerate(analysis_sources):
                in_visible_grid = visible_start <= slot < visible_end
                if in_visible_grid:
                    visible_slot = slot - visible_start
                    r = visible_slot // cols
                    c = visible_slot % cols
                    cell_ox = c * cell_w
                    cell_oy = r * cell_h
                    ox = cell_ox + (cell_w - tile_w) // 2
                    oy = cell_oy + (cell_h - tile_h) // 2
                else:
                    ox = oy = 0

                tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
                if source is None:
                    if in_visible_grid:
                        self._draw_empty_tile(tile)
                        canvas[oy:oy + tile_h, ox:ox + tile_w] = tile
                    continue

                worker = self._grid_workers.get(source)
                if worker is None:
                    if in_visible_grid:
                        self._draw_no_signal_tile(tile, source)
                        canvas[oy:oy + tile_h, ox:ox + tile_w] = tile
                    continue

                with worker["lock"]:
                    frame = worker.get("latest_frame")
                    raw_frame = worker.get("latest_raw_frame")
                    tracks = list(worker.get("tracks", []))
                    track_max_misses = int(worker.get("track_max_misses", 6))
                    max_unknown_tracks = int(worker.get("max_unknown_tracks", 3))

                if frame is None:
                    if in_visible_grid:
                        self._draw_no_signal_tile(tile, source)
                        canvas[oy:oy + tile_h, ox:ox + tile_w] = tile
                    if str(source) == self.viewer_source:
                        self._render_single_no_signal(source)
                    continue

                tile = frame.copy()
                updated_tracks = []
                for t in tracks:
                    tr = t.get("tracker")
                    if tr is None:
                        continue
                    ok, bbox = tr.update(tile)
                    if ok:
                        x, y, w, h = bbox
                        if w > 0.75 * tile_w or h > 0.75 * tile_h:
                            t["misses"] = int(t.get("misses", 0)) + 1
                        elif w < 4 or h < 4:
                            t["misses"] = int(t.get("misses", 0)) + 1
                        else:
                            t["bbox"] = (int(x), int(y), int(w), int(h))
                            t["last_seen"] = now
                            t["misses"] = 0
                    else:
                        t["misses"] = int(t.get("misses", 0)) + 1

                    this_track_max_misses = 3 if t.get("name") == "unknown" else track_max_misses
                    if int(t.get("misses", 0)) <= this_track_max_misses:
                        updated_tracks.append(t)

                # Head detector can sustain known tracks even when CSRT and face detector lose them.
                head_reconfirm_secs = max(3.0, self.detect_every * 5.0 * n_workers)
                updated_tracks = [
                    t for t in updated_tracks
                    if (now - t.get("last_seen", now)) < (unknown_stale_secs if t.get("name") == "unknown" else stale_secs)
                    or (t.get("name") != "unknown" and (now - float(t.get("last_head_t", 0.0))) <= head_reconfirm_secs)
                ]
                updated_tracks = [
                    t for t in updated_tracks
                    if (now - float(t.get("last_detect_t", 0.0))) <= (
                        unknown_reconfirm_secs if t.get("name") == "unknown" else known_reconfirm_secs
                    )
                    or (t.get("name") != "unknown" and (now - float(t.get("last_head_t", 0.0))) <= head_reconfirm_secs)
                ]
                updated_tracks = self._dedupe_tracks(updated_tracks)
                unknown_seen = 0
                limited_tracks = []
                for t in updated_tracks:
                    if t.get("name") == "unknown":
                        unknown_seen += 1
                        if unknown_seen > max_unknown_tracks:
                            continue
                    limited_tracks.append(t)
                updated_tracks = limited_tracks

                with worker["lock"]:
                    worker["tracks"] = updated_tracks
                    worker["latest_tracks"] = [
                        {
                            "name": t.get("name", "unknown"),
                            "best": float(t.get("best", 1.0)),
                            "bbox": [
                                int(t.get("bbox", (0, 0, 0, 0))[0]),
                                int(t.get("bbox", (0, 0, 0, 0))[1]),
                                int(t.get("bbox", (0, 0, 0, 0))[2]),
                                int(t.get("bbox", (0, 0, 0, 0))[3]),
                            ],
                        }
                        for t in updated_tracks
                    ]

                for t in updated_tracks:
                    x, y, w, h = map(int, t.get("bbox", (0, 0, 0, 0)))
                    name = str(t.get("name", "unknown"))
                    best = float(t.get("best", 1.0))
                    x = max(0, int(x))
                    y = max(0, int(y))
                    w = max(1, int(w))
                    h = max(1, int(h))
                    if x + w > tile_w:
                        w = tile_w - x
                    if y + h > tile_h:
                        h = tile_h - y
                    if w <= 0 or h <= 0:
                        continue

                    is_unknown = name == "unknown"
                    color = (0, 0, 255) if is_unknown else (0, 255, 0)
                    cv2.rectangle(tile, (x, y), (x + w, y + h), color, 2)
                    label = "unknown" if is_unknown else name
                    act_label, act_conf = self.get_activity(name) if not is_unknown else (None, 0.0)
                    if act_label:
                        label = f"{name} | {act_label}"
                    cv2.putText(tile, label, (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                    aggregated_tracks.append(
                        {
                            "name": name,
                            "best": float(best),
                            "bbox": [ox + x, oy + y, w, h],
                            "camera_source": str(source),
                            "last_detect_t": float(t.get("last_detect_t", 0.0)),
                            "last_head_t": float(t.get("last_head_t", 0.0)),
                            "activity": act_label,
                            "activity_conf": act_conf,
                        }
                    )

                tile_label = self.source_name_map.get(str(source)) or f"Cam {source}"
                cv2.putText(
                    tile,
                    tile_label,
                    (8, tile_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )
                cv2.rectangle(tile, (0, 0), (tile_w - 1, tile_h - 1), (80, 80, 80), 1)

                snap_due = (now - last_snap_t) >= (1.0 / max(1, self.out_fps))
                known_on_tile = [
                    t.get("name") for t in updated_tracks
                    if t.get("name") and t.get("name") != "unknown"
                ]

                # Build annotated raw frame for footage (native camera resolution)
                # Only do the expensive copy + annotation when footage is actually recording.
                visible = {t.get("name") for t in updated_tracks
                           if t.get("name") and t.get("name") != "unknown"}
                has_active_writer = False
                with self._writers_lock:
                    for vid, rec in self._active_writers.items():
                        if str(rec.get("cam")) == str(source):
                            has_active_writer = True
                            break

                # Build annotated raw frame at native camera resolution.
                # Needed when (a) footage is recording, OR (b) this is the
                # source the user is currently viewing in single mode, OR
                # (c) a visit snapshot is due for a known person on this tile.
                is_viewer_single = str(source) == self.viewer_source
                snap_needs_raw = bool(known_on_tile) and snap_due
                need_annotated_raw = (has_active_writer or is_viewer_single or snap_needs_raw) and raw_frame is not None
                if need_annotated_raw:
                    raw_h, raw_w = raw_frame.shape[:2]
                    sx = raw_w / tile_w
                    sy = raw_h / tile_h
                    annotated_raw = raw_frame.copy()
                    for t in updated_tracks:
                        bx, by, bw, bh = map(int, t.get("bbox", (0, 0, 0, 0)))
                        tname = str(t.get("name", "unknown"))
                        rx, ry = int(bx * sx), int(by * sy)
                        rw, rh = int(bw * sx), int(bh * sy)
                        is_unk = tname == "unknown"
                        col = (0, 0, 255) if is_unk else (0, 255, 0)
                        cv2.rectangle(annotated_raw, (rx, ry), (rx + rw, ry + rh), col, 2)
                        flabel = "unknown" if is_unk else tname
                        f_act, _ = self.get_activity(tname) if not is_unk else (None, 0.0)
                        if f_act:
                            flabel = f"{tname} | {f_act}"
                        font_scale = max(0.5, min(sx, sy) * 0.55)
                        cv2.putText(annotated_raw, flabel, (rx, max(0, ry - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, col, 2)
                    footage_frame = annotated_raw
                elif raw_frame is not None:
                    footage_frame = raw_frame  # no copy needed — just reference for ring buffer
                else:
                    footage_frame = tile

                # Buffer full-resolution snapshot for each known person on this camera.
                # app.py consumes these when opening a new visit. Encoded from the
                # annotated raw frame (native camera resolution) so saved visit
                # screenshots match the camera's true resolution.
                if known_on_tile and snap_due:
                    snap_src = footage_frame if footage_frame is not None else tile
                    ok_snap, snap_buf = cv2.imencode(
                        ".jpg", snap_src, [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                    )
                    if ok_snap:
                        snap_bytes = snap_buf.tobytes()
                        with self._snapshot_lock:
                            for pname in known_on_tile:
                                self._snapshot_buf[(pname, str(source))] = snap_bytes
                    last_snap_t = now

                # Push to rolling frame ring buffer + active writers at native resolution
                self._push_frame_to_ring(source, footage_frame)
                if has_active_writer:
                    self._feed_active_writers(source, footage_frame, visible)

                # Encode the annotated full-res frame for single-view display.
                # Throttled to out_fps to keep CPU encode work bounded.
                if (is_viewer_single and need_annotated_raw and
                        (now - last_single_encode_t) >= (1.0 / max(1, self.out_fps))):
                    last_single_encode_t = now
                    ok_s, buf_s = cv2.imencode(
                        ".jpg", footage_frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)],
                    )
                    if ok_s:
                        with self._lock:
                            self._latest_single_jpeg = buf_s.tobytes()

                if in_visible_grid:
                    canvas[oy:oy + tile_h, ox:ox + tile_w] = tile

            # Fill any visible slots that fall past the end of the source
            # list with an "empty" tile (e.g. when paging beyond the last
            # camera with a layout that has more slots than remaining
            # sources).
            total_visible = rows * cols
            n_sources = len(analysis_sources)
            for vis in range(total_visible):
                abs_slot = visible_start + vis
                if abs_slot < n_sources:
                    continue
                er = vis // cols
                ec = vis % cols
                eox = ec * cell_w + (cell_w - tile_w) // 2
                eoy = er * cell_h + (cell_h - tile_h) // 2
                empty_tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
                self._draw_empty_tile(empty_tile)
                canvas[eoy:eoy + tile_h, eox:eox + tile_w] = empty_tile

            # QR scan round-robin (skip None/empty slots)
            active_sources = [s for s in self._grid_sources if s is not None]
            if active_sources and (now - self._last_qr_scan_t) >= self.qr_scan_every:
                self._last_qr_scan_t = now
                source = active_sources[self._grid_qr_rr_idx % len(active_sources)]
                self._grid_qr_rr_idx += 1
                worker = self._grid_workers.get(source)
                if worker is not None:
                    with worker["lock"]:
                        qr_frame = worker.get("latest_frame")
                    if qr_frame is not None:
                        try:
                            data, _pts, _ = self._qr_detector.detectAndDecode(qr_frame)
                            data = (data or "").strip()
                            if data:
                                with self._lock:
                                    self._latest_qr = data
                                    self._latest_qr_t = now
                        except Exception:
                            pass

            if now - last_encode_t >= (1.0 / max(1, self.out_fps)):
                last_encode_t = now
                ok, buf = cv2.imencode(
                    ".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)]
                )
                if ok:
                    with self._lock:
                        self._latest_jpeg = buf.tobytes()
                        self._latest_tracks = aggregated_tracks

            time.sleep(0.001)

    def _main_loop(self):
        last_sent_t = 0.0
        last_encode_t = 0.0
        last_snap_t = 0.0
        stale_secs = 2.0
        unknown_stale_secs = 0.7
        unknown_reconfirm_secs = max(1.5, self.detect_every * 1.5)
        known_reconfirm_secs = max(2.0, self.detect_every * 2.0)
        max_detection_lag = max(1.5, self.detect_every * 4.0)

        while not self.stop_evt.is_set():
            if self._cap is None:
                time.sleep(0.05)
                continue

            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            now = time.monotonic()
            # --- QR scan (throttled) ---
            if (now - self._last_qr_scan_t) >= self.qr_scan_every:
                self._last_qr_scan_t = now
                try:
                    data, pts, _ = self._qr_detector.detectAndDecode(frame)
                    data = (data or "").strip()
                    if data:
                        with self._lock:
                            self._latest_qr = data
                            self._latest_qr_t = now
                except Exception:
                    # QR scan is best-effort; ignore errors
                    pass

            Hf, Wf = frame.shape[:2]

            # 1) update trackers
            new_tracks = []
            for t in self.tracks:
                ok, bbox = t["tracker"].update(frame)
                if ok:
                    x, y, w, h = bbox
                    # kill drifted/exploded trackers
                    if w > 0.75 * Wf or h > 0.75 * Hf:
                        continue
                    if w < 20 or h < 20:
                        continue
                    t["bbox"] = bbox
                    new_tracks.append(t)
            self.tracks = new_tracks

            # 2) send to detector (immediate if no tracks)
            if (not self.tracks) or (now - last_sent_t >= self.detect_every):
                last_sent_t = now
                self._push_latest(self.in_q, (now, frame.copy()))

            # 3) consume detections, update tracks by IoU
            try:
                det_out = self.out_q.get_nowait()
                head_bboxes_from_det = []
                if isinstance(det_out, tuple) and len(det_out) == 3:
                    det_t, results, head_bboxes_from_det = det_out
                elif isinstance(det_out, tuple) and len(det_out) == 2:
                    det_t, results = det_out
                else:
                    det_t, results = now, det_out

                UNKNOWN_TO_FORGET = 3  # how many consecutive "unknown" hits before we downgrade a known track
                unknown_stale_secs = max(1.2, self.detect_every * 1.2)  # must be >= detect_every to avoid flicker
                # Ignore stale detector results only when we already have active tracks.
                # This avoids losing bootstrap detections on slower hardware.
                det_age = now - float(det_t)
                if self.tracks and det_age > max_detection_lag:
                    results = []
                results = self._dedupe_detections(results)
                matched_track_idx = set()
                # Process larger faces first for more stable assignment.
                results = sorted(results, key=lambda r: int(r[2]) * int(r[3]), reverse=True)

                for x, y, w, h, name, best in results:
                    det_bbox = (int(x), int(y), int(w), int(h))

                    # Find best track match (IoU first, center-distance fallback).
                    best_i = -1
                    best_key = None
                    for i, t in enumerate(self.tracks):
                        if i in matched_track_idx:
                            continue
                        t_name = t.get("name", "unknown")
                        compatible = (t_name == name) or (t_name == "unknown") or (name == "unknown")
                        if not compatible:
                            continue

                        t_bbox = t["bbox"]
                        center_dist = self._center_dist_norm(t_bbox, det_bbox)
                        score = iou(t["bbox"], det_bbox)
                        motion_match = (
                            (t_name == name and name != "unknown" and center_dist < 0.75)
                            or (((t_name == "unknown") ^ (name == "unknown")) and center_dist < 0.50)
                            or (t_name == "unknown" and name == "unknown" and center_dist < 0.40)
                        )
                        if score <= 0.3 and not motion_match:
                            continue

                        # Prefer higher IoU, then known identity, then smaller center distance.
                        cand_key = (score, 1 if t_name != "unknown" else 0, -center_dist)
                        if (best_key is None) or (cand_key > best_key):
                            best_key = cand_key
                            best_i = i

                    if best_i >= 0:
                        t = self.tracks[best_i]
                        matched_track_idx.add(best_i)

                        # always refresh bbox/tracker/last_seen
                        t["bbox"] = det_bbox
                        t["tracker"] = create_tracker(self.tracker_type)
                        t["tracker"].init(frame, det_bbox)
                        t["updated"] = True
                        t["last_seen"] = now
                        t["last_detect_t"] = now
                        t["last_head_t"] = now   # face confirmed ⇒ head confirmed too

                        # label smoothing: unknown doesn't immediately clobber a known identity
                        if name == "unknown":
                            if t.get("name") != "unknown":
                                t["unknown_hits"] = t.get("unknown_hits", 0) + 1
                                if t["unknown_hits"] >= UNKNOWN_TO_FORGET:
                                    t["name"] = "unknown"
                                    t["best"] = float(best)
                            else:
                                t["best"] = float(best)
                        else:
                            t["name"] = name
                            t["best"] = float(best)
                            t["unknown_hits"] = 0
                    else:
                        # If this detection is very close to an existing track, treat it
                        # as duplicate noise instead of creating a second box.
                        near_existing = False
                        for t in self.tracks:
                            t_name = t.get("name", "unknown")
                            compatible = (t_name == name) or (t_name == "unknown") or (name == "unknown")
                            if not compatible:
                                continue
                            ov = iou(t["bbox"], det_bbox)
                            cd = self._center_dist_norm(t["bbox"], det_bbox)
                            if ov > 0.45 or cd < 0.32:
                                near_existing = True
                                break
                        if near_existing:
                            continue

                        # new track
                        tr = create_tracker(self.tracker_type)
                        tr.init(frame, det_bbox)
                        self.tracks.append(
                            {
                                "tracker": tr,
                                "name": name,
                                "best": float(best),
                                "bbox": det_bbox,
                                "updated": True,
                                "last_seen": now,
                                "last_detect_t": now,
                                "last_head_t": now,
                                "unknown_hits": 0 if name != "unknown" else 1,
                            }
                        )

                # drop stale tracks (head detector can sustain known tracks)
                head_reconfirm_secs = max(3.0, self.detect_every * 5.0)
                self.tracks = [
                    t for t in self.tracks
                    if (now - t.get("last_seen", now)) < (unknown_stale_secs if t.get("name") == "unknown" else stale_secs)
                    or (t.get("name") != "unknown" and (now - float(t.get("last_head_t", 0.0))) <= head_reconfirm_secs)
                ]
                # drop tracks not reconfirmed by face OR head detector (ghost box prevention)
                self.tracks = [
                    t for t in self.tracks
                    if (now - float(t.get("last_detect_t", now))) <= (
                        unknown_reconfirm_secs if t.get("name") == "unknown" else known_reconfirm_secs
                    )
                    or (t.get("name") != "unknown" and (now - float(t.get("last_head_t", 0.0))) <= head_reconfirm_secs)
                ]
                self.tracks = self._dedupe_tracks(self.tracks)

                # --- Head detection: sustain tracks when face not visible ---
                self._match_heads_to_tracks(self.tracks, head_bboxes_from_det, now)

                # --- Auto-capture unknowns ---
                if self.auto_capture_enabled:
                    for t in self.tracks:
                        if not t.get("updated"):
                            continue
                        bbox = t.get("bbox", (0, 0, 0, 0))
                        if t.get("name") == "unknown":
                            captured = self._try_capture_unknown(t, frame, now)
                            if captured:
                                print(f"[single-cam] auto-captured {captured}")
                        else:
                            # Top up samples for any recognised person.
                            self._try_capture_more_for_known(
                                t, frame, bbox, now,
                                camera_source=self.cam_index,
                            )
                    self._cleanup_unknown_pending({id(t) for t in self.tracks})

                # --- Action detection (enqueue to async thread, throttled) ---
                # Only enqueue when face detector recently confirmed the person;
                # head-detector-only or tracker-only boxes (e.g. back of head) are skipped.
                if self.activity_enabled:
                    face_fresh_secs = self.detect_every * 2.0
                    for t in self.tracks:
                        name = t.get("name", "unknown")
                        if name == "unknown":
                            continue
                        if (now - float(t.get("last_detect_t", 0.0))) > face_fresh_secs:
                            continue
                        last_act_t = float(t.get("_last_activity_t", 0.0))
                        if (now - last_act_t) < self.activity_detect_every:
                            continue
                        bbox = t.get("bbox", (0, 0, 0, 0))
                        try:
                            self._activity_q.put_nowait((frame.copy(), bbox, name))
                        except Full:
                            pass
                        t["_last_activity_t"] = now

            except Empty:
                pass

            # 4) draw
            for t in self.tracks:
                x, y, w, h = map(int, t["bbox"])

                is_unknown = (t["name"] == "unknown")
                color = (0, 0, 255) if is_unknown else (0, 255, 0)  # red for unknown, green for known

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                label = "unknown" if is_unknown else t['name']
                act_label, _ = self.get_activity(t['name']) if not is_unknown else (None, 0.0)
                if act_label:
                    label = f"{t['name']} | {act_label}"
                cv2.putText(frame, label, (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,)

            # 4b) buffer snapshot for each known person (single-camera mode)
            # Throttled to out_fps to avoid redundant JPEG encodes every frame.
            snap_due = (now - last_snap_t) >= (1.0 / max(1, self.out_fps))
            known_on_frame = [
                t["name"] for t in self.tracks
                if t.get("name") and t["name"] != "unknown"
            ]
            if known_on_frame and snap_due:
                ok_snap, snap_buf = cv2.imencode(
                    ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                )
                if ok_snap:
                    snap_bytes = snap_buf.tobytes()
                    cam_src = str(self.cam_index)
                    with self._snapshot_lock:
                        for pname in known_on_frame:
                            self._snapshot_buf[(pname, cam_src)] = snap_bytes
                last_snap_t = now

            # 4c) push frame to rolling ring buffer + active writers (single-camera)
            self._push_frame_to_ring(self.cam_index, frame)
            visible = {t["name"] for t in self.tracks
                       if t.get("name") and t["name"] != "unknown"}
            has_active_writer = bool(self._active_writers)
            if has_active_writer:
                self._feed_active_writers(self.cam_index, frame, visible)

            # 5) encode at limited FPS
            if now - last_encode_t >= (1.0 / max(1, self.out_fps)):
                last_encode_t = now
                ok, buf = cv2.imencode(
                    ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)]
                )
                if ok:
                    # Build track list outside lock to avoid nested lock acquisition
                    pub_tracks = [
                        {
                            "name": t["name"],
                            "best": float(t["best"]),
                            "bbox": [int(t["bbox"][0]), int(t["bbox"][1]), int(t["bbox"][2]), int(t["bbox"][3])],
                            "last_detect_t": float(t.get("last_detect_t", 0.0)),
                            "last_head_t": float(t.get("last_head_t", 0.0)),
                            "activity": self.get_activity(t["name"])[0],
                            "activity_conf": self.get_activity(t["name"])[1],
                        }
                        for t in self.tracks
                    ]
                    jpeg_bytes = buf.tobytes()
                    with self._lock:
                        self._latest_jpeg = jpeg_bytes
                        self._latest_tracks = pub_tracks
