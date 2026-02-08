import cv2
import os
import numpy as np
from deepface import DeepFace
import time
import threading
from queue import Full, Queue, Empty

KNOWN_DIR = "faces"
THRESHOLD = 0.4
DETECTOR = "opencv"
MODEL = "Facenet"
ALLOWED_EXTS = (".jpg", ".png", ".jpeg", ".webp")
DETECT_EVERY = 1.0          # seconds
DETECT_SCALE = 0.5          # detect on smaller frame (0.5 = 50% size). try 0.4–0.7
TRACKER_TYPE = "CSRT"        # "KCF" fast, "CSRT" slower but steadier

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
        raise RuntimeError("OpenCV tracker not available. Install opencv-contrib-python (and uninstall opencv-python if both are installed).")
    return tr

# ---- Precompute known embeddings once ----
def l2norm(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-10)

known_embeddings = []

for person in os.listdir(KNOWN_DIR):
    person_dir = os.path.join(KNOWN_DIR, person)
    if not os.path.isdir(person_dir):
        continue

    embs = []
    for file in os.listdir(person_dir):
        if not file.lower().endswith(ALLOWED_EXTS):
            continue

        path = os.path.join(person_dir, file)
        try:
            faces = DeepFace.extract_faces(
                img_path=path,
                detector_backend=DETECTOR,
                enforce_detection=True
            )
            if not faces:
                continue

            face_img = faces[0]["face"]  # cropped (often aligned)
            rep = DeepFace.represent(img_path=face_img, model_name=MODEL, detector_backend="skip", enforce_detection=False)
            if rep and "embedding" in rep[0]:
                embs.append(l2norm(rep[0]["embedding"]))
        except Exception:
            continue

    if embs:
        template = l2norm(np.mean(np.asarray(embs), axis=0))
        known_embeddings.append((person, template))

# ---- Webcam ----
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Starting webcam — press Q or ESC to quit")

# ---- Background worker setup ----
in_q = Queue(maxsize=1)   # holds latest frame for detection
out_q = Queue(maxsize=1)  # holds latest detection+ID results
stop_evt = threading.Event()

def detector_worker():
    while not stop_evt.is_set():
        try:
            frame_full = in_q.get(timeout=0.1)  # full-res
        except Empty:
            continue

        results = []

        # 1) detect on downscaled frame
        small = cv2.resize(frame_full, None, fx=DETECT_SCALE, fy=DETECT_SCALE)
        try:
            faces = DeepFace.extract_faces(img_path=small, detector_backend=DETECTOR, enforce_detection=True)
        except Exception:
            faces = []

        H, W = frame_full.shape[:2]
        inv = 1.0 / DETECT_SCALE

        for face in faces:
            fa = face["facial_area"]
            x = int(fa["x"] * inv)
            y = int(fa["y"] * inv)
            w = int(fa["w"] * inv)
            h = int(fa["h"] * inv)

            # clip to full frame
            x = max(0, x); y = max(0, y)
            w = max(1, min(w, W - x))
            h = max(1, min(h, H - y))

            face_img = face.get("face")  # <- DeepFace's cropped face
            if face_img is None or getattr(face_img, "size", 0) == 0:
                continue

            try:
                rep_live = DeepFace.represent(img_path=face_img, model_name=MODEL, detector_backend="skip", enforce_detection=False)
                if not rep_live:
                    continue
                emb_live = l2norm(rep_live[0]["embedding"])
            except Exception:
                continue

            # 3) match
            name = "Unknown"
            best = 1.0
            for known_name, emb_known in known_embeddings:
                d = cosine_distance(emb_live, emb_known)
                if d < best:
                    best = d
                    name = known_name

            if best > THRESHOLD:
                name = "Unknown"

            results.append((x, y, w, h, name, best))

        # publish latest
        while True:
            try:
                out_q.get_nowait()
            except Empty:
                break
        out_q.put(results)

worker = threading.Thread(target=detector_worker, daemon=True)
worker.start()

# ---- Tracking state ----
tracks = []
last_sent_t = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1) Update trackers every frame
    new_tracks = []
    for t in tracks:
        ok, bbox = t["tracker"].update(frame)
        if ok:
            t["bbox"] = bbox
            new_tracks.append(t)
    tracks = new_tracks

    # 2) Send a frame to the detector once per second (non-blocking)
    def push_latest(q, item):
        try:
            q.put_nowait(item)
        except Full:
            try:
                q.get_nowait()
            except Empty:
                pass
            q.put_nowait(item)

    now = time.monotonic()
    if now - last_sent_t >= DETECT_EVERY:
        last_sent_t = now
        push_latest(in_q, frame.copy()) 

    # 3) If worker produced new results, rebuild trackers (fast)
    try:
        results = out_q.get_nowait()

        # mark all tracks as not-updated this cycle
        for t in tracks:
            t["updated"] = False

        for x, y, w, h, name, best in results:
            det_bbox = (int(x), int(y), int(w), int(h))

            # find best matching existing track
            best_iou = 0.0
            best_idx = -1
            for i, t in enumerate(tracks):
                score = iou(t["bbox"], det_bbox)
                if score > best_iou:
                    best_iou = score
                    best_idx = i

            if best_iou > 0.3 and best_idx >= 0:
                # update existing track (re-init tracker on refreshed bbox)
                t = tracks[best_idx]
                t["name"] = name
                t["best"] = best
                t["bbox"] = det_bbox
                t["tracker"] = create_tracker(TRACKER_TYPE)
                t["tracker"].init(frame, det_bbox)
                t["updated"] = True
            else:
                # new face => new track
                tr = create_tracker(TRACKER_TYPE)
                tr.init(frame, det_bbox)
                tracks.append({"tracker": tr, "name": name, "best": best, "bbox": det_bbox, "updated": True})

        # optionally drop stale tracks that weren’t matched for a while
        tracks = [t for t in tracks if t.get("updated", True)]

    except Empty:
        pass

    # 4) Draw tracks every frame (cheap)
    for t in tracks:
        x, y, w, h = t["bbox"]
        x, y, w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{t['name']} ({t['best']:.2f})", (x, max(0, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Face Recognition (Smooth)", frame)
    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
        break

stop_evt.set()
cap.release()
cv2.destroyAllWindows()
print("Exited cleanly")