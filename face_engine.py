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
        self._worker_t = None
        self._main_t = None
        self._lock = threading.Lock()
        self._latest_jpeg = None
        self._latest_tracks = []
        self._running = False

    # ---------- public ----------
    def start(self):
        if self._running:
            return

        self.stop_evt.clear()
        self.reload_faces()

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
                frame_full = self.in_q.get(timeout=0.1)
            except Empty:
                continue

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

                # clip
                x = max(0, x)
                y = max(0, y)
                w = max(1, min(w, W - x))
                h = max(1, min(h, H - y))

                # filters to avoid tiny/huge junk
                if w < 90 or h < 90:
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

                # embed + match
                try:
                    rep_live = DeepFace.represent(
                        img_path=face_img,
                        model_name=self.model,
                        detector_backend="skip",
                        enforce_detection=False,
                    )
                    if not rep_live:
                        continue
                    emb_live = l2norm(rep_live[0]["embedding"])
                except Exception:
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

            # publish latest
            while True:
                try:
                    self.out_q.get_nowait()
                except Empty:
                    break
            self.out_q.put(results)

    def _main_loop(self):
        last_sent_t = 0.0
        last_encode_t = 0.0
        stale_secs = 2.0
        unknown_stale_secs = 0.7

        while not self.stop_evt.is_set():
            if self._cap is None:
                time.sleep(0.05)
                continue

            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            now = time.monotonic()
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
                self._push_latest(self.in_q, frame.copy())

            # 3) consume detections, update tracks by IoU
            try:
                results = self.out_q.get_nowait()

                UNKNOWN_TO_FORGET = 3  # how many consecutive "unknown" hits before we downgrade a known track
                unknown_stale_secs = max(1.2, self.detect_every * 1.2)  # must be >= detect_every to avoid flicker

                for x, y, w, h, name, best in results:
                    det_bbox = (int(x), int(y), int(w), int(h))

                    # find best track match (no type-gating)
                    best_i = -1
                    best_score = 0.0
                    for i, t in enumerate(self.tracks):
                        score = iou(t["bbox"], det_bbox)
                        if score > best_score:
                            best_score = score
                            best_i = i

                    if best_score > 0.3 and best_i >= 0:
                        t = self.tracks[best_i]

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
