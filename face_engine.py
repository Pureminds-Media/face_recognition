import os
import time
import threading
from queue import Queue, Empty, Full
import cv2
import numpy as np
from deepface import DeepFace


def l2norm(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-10)


def cosine_distance(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
    return 1.0 - float(np.dot(a, b) / denom)


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
        self._grid_stop_evt = threading.Event()
        self._grid_render_t = None
        self._grid_layout = (3, 2)  # rows, cols
        self._grid_qr_rr_idx = 0
        self._worker_t = None
        self._main_t = None
        self._lock = threading.Lock()
        self._latest_jpeg = None
        self._latest_tracks = []
        self._running = False
        self._qr_detector = cv2.QRCodeDetector()
        self.qr_scan_every = 0.5  # seconds (2 scans/sec). Increase if CPU is high.
        self._last_qr_scan_t = 0.0
        self._latest_qr = None  # last decoded QR string
        self._latest_qr_t = 0.0 # monotonic time of last decoded QR

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

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap = cap

            self._worker_t = threading.Thread(target=self._detector_worker, daemon=True)
            self._worker_t.start()

            self._main_t = threading.Thread(target=self._main_loop, daemon=True)
            self._main_t.start()

        self._running = True

    def stop(self):
        if not self._running:
            return

        # Signal threads to exit
        self._running = False
        self.stop_evt.set()

        # Release camera early to unblock reads
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._cleanup_grid_mode()

        # Join threads
        if self._worker_t is not None:
            self._worker_t.join(timeout=2.0)
            self._worker_t = None

        if self._main_t is not None:
            self._main_t.join(timeout=2.0)
            self._main_t = None

        # Clear queues
        with self.in_q.mutex:
            self.in_q.queue.clear()
        with self.out_q.mutex:
            self.out_q.queue.clear()

        # Clear published state
        with self._lock:
            self._latest_jpeg = None
            self._latest_tracks = []

        self.tracks = []


    def is_running(self):
        return self._running

    def get_jpeg(self):
        with self._lock:
            return self._latest_jpeg

    def get_tracks(self):
        with self._lock:
            return list(self._latest_tracks)
    
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
                    else:
                        video_tracks.append(
                            {
                                "bbox": det_bbox,
                                "name": name,
                                "best": float(best),
                                "matched": True,
                                "misses": 0,
                                "unknown_hits": 0 if name != "unknown" else 1,
                            }
                        )

                kept_tracks = []
                for t in video_tracks:
                    if not t.get("matched"):
                        t["misses"] = int(t.get("misses", 0)) + 1
                    if int(t.get("misses", 0)) <= track_max_misses:
                        kept_tracks.append(t)
                video_tracks = self._dedupe_tracks(kept_tracks, iou_thresh=0.50, center_dist_thresh=0.65)

                for t in video_tracks:
                    x, y, w, h = map(int, t.get("bbox", (0, 0, 0, 0)))
                    name = t.get("name", "unknown")
                    best = float(t.get("best", 1.0))
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
                    label = "unknown" if is_unknown else f"{name} ({best:.2f})"
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
        """Drop overlapping duplicate tracks for the same person."""
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
                # Use center distance as a fallback only when identities are compatible.
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

        kept = []
        for t in sorted(tracks, key=_rank, reverse=True):
            duplicate = False
            for k in kept:
                if _can_merge(t, k):
                    duplicate = True
                    break
            if not duplicate:
                kept.append(t)
        return kept

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
                        enforce_detection=True,
                        align=True,
                    )
                    if not faces:
                        continue
                    face_img = faces[0].get("face")
                    if face_img is None:
                        continue
                    rep = DeepFace.represent(
                        img_path=face_img,
                        model_name=self.model,
                        detector_backend="skip",
                        enforce_detection=False,
                    )
                    if rep and "embedding" in rep[0]:
                        embs.append(l2norm(rep[0]["embedding"]))
                except Exception as e:
                    print(f"Error processing {file}: {e}")
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
        except Exception:
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
            except Exception:
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

            # publish latest
            while True:
                try:
                    self.out_q.get_nowait()
                except Empty:
                    break
            self.out_q.put((float(frame_t), results))

    def _is_grid_mode(self):
        return isinstance(self.cam_index, str) and self.cam_index.startswith("grid:")

    @staticmethod
    def _parse_grid_sources(cam_index):
        if not (isinstance(cam_index, str) and cam_index.startswith("grid:")):
            return []
        raw = cam_index.split(":", 1)[1].strip()
        if not raw:
            return []

        out = []
        for token in raw.split(","):
            t = token.strip()
            if not t:
                continue
            if t.isdigit():
                out.append(int(t))
            else:
                out.append(t)
        return out

    def _start_grid_mode(self):
        sources = self._parse_grid_sources(self.cam_index)
        if not sources:
            raise RuntimeError("No camera sources configured for grid mode")

        rows, cols = self._grid_layout
        tile_w = max(1, int(self.width // cols))
        tile_h = max(1, int(self.height // rows))
        max_slots = 6
        workers = {}
        ordered_sources = []
        self._grid_stop_evt.clear()
        self._grid_qr_rr_idx = 0

        for source in sources[:max_slots]:
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                try:
                    cap.release()
                except Exception:
                    pass
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, tile_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, tile_h)
            if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass

            worker = {
                "source": source,
                "cap": cap,
                "lock": threading.Lock(),
                "stop_evt": threading.Event(),
                "frame_q": Queue(maxsize=1),
                "latest_frame": None,
                "latest_frame_t": 0.0,
                "last_good_frame_t": 0.0,
                "tracks": [],
                "latest_tracks": [],
                "last_det_t": 0.0,
                "track_max_misses": 6,
                "bbox_smooth_alpha": 0.55,
                "max_unknown_tracks": 3,
                "capture_thread": None,
                "detect_thread": None,
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
            workers[str(source)] = worker
            ordered_sources.append(str(source))

        if not workers:
            self._cleanup_grid_mode()
            raise RuntimeError("Cannot open any camera for grid mode")

        self._grid_workers = workers
        self._grid_sources = ordered_sources

        for source in self._grid_sources:
            w = self._grid_workers[source]
            w["capture_thread"].start()
            w["detect_thread"].start()

        self._grid_render_t = threading.Thread(target=self._grid_render_loop, daemon=True)
        self._grid_render_t.start()

    def _cleanup_grid_mode(self):
        self._grid_stop_evt.set()

        for w in self._grid_workers.values():
            try:
                w["stop_evt"].set()
            except Exception:
                pass

        for w in self._grid_workers.values():
            cap = w.get("cap")
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

        for w in self._grid_workers.values():
            t = w.get("capture_thread")
            if t is not None:
                t.join(timeout=2.0)
            t = w.get("detect_thread")
            if t is not None:
                t.join(timeout=2.0)

        if self._grid_render_t is not None:
            self._grid_render_t.join(timeout=2.0)
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

            try:
                frame = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            except Exception:
                time.sleep(0.01)
                continue

            with worker["lock"]:
                worker["latest_frame"] = frame
                worker["latest_frame_t"] = now
                worker["last_good_frame_t"] = now

            self._push_latest(worker["frame_q"], (now, frame.copy()))

    def _grid_detect_loop(self, worker):
        UNKNOWN_TO_FORGET = 3
        detect_period = max(0.25, float(self.detect_every))
        while not self.stop_evt.is_set() and not self._grid_stop_evt.is_set() and not worker["stop_evt"].is_set():
            try:
                _frame_t, frame = worker["frame_q"].get(timeout=0.1)
            except Empty:
                continue

            now = time.monotonic()
            with worker["lock"]:
                last_det_t = float(worker.get("last_det_t", 0.0))
            if (now - last_det_t) < detect_period:
                continue

            try:
                dets = self._detect_and_match_faces(
                    frame,
                    min_face_size=20,
                    keep_unembedded_unknown=True,
                )
                dets = self._dedupe_detections(dets)
            except Exception:
                dets = []

            with worker["lock"]:
                tracks = list(worker.get("tracks", []))
                bbox_smooth_alpha = float(worker.get("bbox_smooth_alpha", 0.55))
                track_max_misses = int(worker.get("track_max_misses", 6))
                max_unknown_tracks = int(worker.get("max_unknown_tracks", 3))
                matched_track_idx = set()
                Hf, Wf = frame.shape[:2]

                # Match larger detections first for more stable assignment.
                dets = sorted(dets, key=lambda r: int(r[2]) * int(r[3]), reverse=True)
                for x, y, w, h, name, best in dets:
                    det_bbox = (int(x), int(y), int(w), int(h))
                    if name == "unknown":
                        # Suppress noisy unknowns (tiny boxes and edge-hugging false positives).
                        if int(w) < 20 or int(h) < 20:
                            continue
                        margin_x = max(3, int(0.01 * Wf))
                        margin_y = max(3, int(0.01 * Hf))
                        if (x <= margin_x) or (y <= margin_y) or ((x + w) >= (Wf - margin_x)) or ((y + h) >= (Hf - margin_y)):
                            continue
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
                                "misses": 0,
                                "unknown_hits": 0 if name != "unknown" else 1,
                            }
                        )

                for i, t in enumerate(tracks):
                    if i not in matched_track_idx:
                        t["misses"] = int(t.get("misses", 0)) + 1

                tracks = [t for t in tracks if int(t.get("misses", 0)) <= track_max_misses]
                tracks = self._dedupe_tracks(tracks)
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
                    }
                    for t in tracks
                ]
                worker["last_det_t"] = now

    def _grid_render_loop(self):
        rows, cols = self._grid_layout
        tile_w = max(1, int(self.width // cols))
        tile_h = max(1, int(self.height // rows))
        grid_w = tile_w * cols
        grid_h = tile_h * rows
        last_encode_t = 0.0
        stale_secs = 1.6
        unknown_stale_secs = max(0.8, self.detect_every * 1.0)
        # Unknown tracks must be periodically reconfirmed by detector; tracker-only
        # unknown boxes are aggressively expired to avoid ghost boxes.
        unknown_reconfirm_secs = max(1.2, self.detect_every * 2.8)

        while not self.stop_evt.is_set() and not self._grid_stop_evt.is_set():
            now = time.monotonic()
            canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
            aggregated_tracks = []

            for slot in range(rows * cols):
                r = slot // cols
                c = slot % cols
                ox = c * tile_w
                oy = r * tile_h

                tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
                if slot >= len(self._grid_sources):
                    self._draw_empty_tile(tile)
                    canvas[oy:oy + tile_h, ox:ox + tile_w] = tile
                    continue

                source = self._grid_sources[slot]
                worker = self._grid_workers.get(source)
                if worker is None:
                    self._draw_no_signal_tile(tile, source)
                    canvas[oy:oy + tile_h, ox:ox + tile_w] = tile
                    continue

                with worker["lock"]:
                    frame = worker.get("latest_frame")
                    tracks = list(worker.get("tracks", []))
                    track_max_misses = int(worker.get("track_max_misses", 6))
                    max_unknown_tracks = int(worker.get("max_unknown_tracks", 3))

                if frame is None:
                    self._draw_no_signal_tile(tile, source)
                    canvas[oy:oy + tile_h, ox:ox + tile_w] = tile
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
                        elif w < 12 or h < 12:
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

                updated_tracks = [
                    t for t in updated_tracks
                    if (now - t.get("last_seen", now)) < (unknown_stale_secs if t.get("name") == "unknown" else stale_secs)
                ]
                updated_tracks = [
                    t for t in updated_tracks
                    if (t.get("name") != "unknown")
                    or ((now - float(t.get("last_detect_t", 0.0))) <= unknown_reconfirm_secs)
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
                    label = "unknown" if is_unknown else f"{name} ({float(best):.2f})"
                    cv2.putText(tile, label, (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                    aggregated_tracks.append(
                        {
                            "name": name,
                            "best": float(best),
                            "bbox": [ox + x, oy + y, w, h],
                            "camera_source": str(source),
                        }
                    )

                cv2.putText(
                    tile,
                    f"Cam {source}",
                    (8, tile_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )
                cv2.rectangle(tile, (0, 0), (tile_w - 1, tile_h - 1), (80, 80, 80), 1)
                canvas[oy:oy + tile_h, ox:ox + tile_w] = tile

            if self._grid_sources and (now - self._last_qr_scan_t) >= self.qr_scan_every:
                self._last_qr_scan_t = now
                source = self._grid_sources[self._grid_qr_rr_idx % len(self._grid_sources)]
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
        stale_secs = 2.0
        unknown_stale_secs = 0.7
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
                if (
                    isinstance(det_out, tuple)
                    and len(det_out) == 2
                ):
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
                                "unknown_hits": 0 if name != "unknown" else 1,
                            }
                        )

                # drop stale tracks
                self.tracks = [
                    t for t in self.tracks
                    if (now - t.get("last_seen", now)) < (unknown_stale_secs if t.get("name") == "unknown" else stale_secs)
                ]
                self.tracks = self._dedupe_tracks(self.tracks)

            except Empty:
                pass

            # 4) draw
            for t in self.tracks:
                x, y, w, h = map(int, t["bbox"])

                is_unknown = (t["name"] == "unknown")
                color = (0, 0, 255) if is_unknown else (0, 255, 0)  # red for unknown, green for known

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                label = "unknown" if is_unknown else f"{t['name']} ({t['best']:.2f})"
                cv2.putText(frame, label, (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,)

            # 5) encode at limited FPS
            if now - last_encode_t >= (1.0 / max(1, self.out_fps)):
                last_encode_t = now
                ok, buf = cv2.imencode(
                    ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)]
                )
                if ok:
                    with self._lock:
                        self._latest_jpeg = buf.tobytes()
                        self._latest_tracks = [
                            {
                                "name": t["name"],
                                "best": float(t["best"]),
                                "bbox": [int(t["bbox"][0]), int(t["bbox"][1]), int(t["bbox"][2]), int(t["bbox"][3])],
                            }
                            for t in self.tracks
                        ]
