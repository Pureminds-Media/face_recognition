"""YOLOv8-nano head detector using ONNX Runtime.

Detects heads regardless of face visibility (front, back, profile, top).
Trained on SCUT-HEAD dataset via YOLOv8 nano architecture.

Usage::

    hd = HeadDetector()
    heads = hd.detect(frame)  # list of (x, y, w, h, confidence)
"""

from __future__ import annotations

import logging
import os
import time

import cv2
import numpy as np

log = logging.getLogger(__name__)

_DEFAULT_MODEL = os.path.join(
    os.path.dirname(__file__), "models", "head-yolov8n-onnx", "model.onnx"
)

# YOLO input size the model was exported with
_INPUT_SIZE = 640


def _ensure_cuda_libs():
    """Pre-load NVIDIA shared libraries so ``onnxruntime`` can find CUDA.

    When running inside a venv with ``nvidia-*`` pip packages the shared
    objects live in ``site-packages/nvidia/*/lib/`` which is *not* on
    ``LD_LIBRARY_PATH`` by default.  Loading them with ``ctypes.CDLL``
    and ``RTLD_GLOBAL`` makes them visible to ``onnxruntime`` when it
    tries to initialise ``CUDAExecutionProvider``.
    """
    import ctypes
    import glob
    import site

    sp_dirs = site.getsitepackages()
    for sp in sp_dirs:
        nv_base = os.path.join(sp, "nvidia")
        if not os.path.isdir(nv_base):
            continue
        for lib_dir in sorted(glob.glob(os.path.join(nv_base, "*/lib"))):
            for so in sorted(glob.glob(os.path.join(lib_dir, "*.so*"))):
                try:
                    ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


class HeadDetector:
    """YOLOv8-nano head detector running on ONNX Runtime (GPU preferred).

    Parameters
    ----------
    model_path:
        Path to the ``.onnx`` model file.
    conf_threshold:
        Minimum detection confidence.
    iou_threshold:
        IoU threshold for Non-Maximum Suppression.
    """

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL,
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.50,
    ):
        self._model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self._session = None  # lazy
        self._input_name: str = ""

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------
    def _ensure_loaded(self):
        if self._session is not None:
            return

        if not os.path.isfile(self._model_path):
            raise FileNotFoundError(
                f"Head detection model not found: {self._model_path}"
            )

        _ensure_cuda_libs()

        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        t0 = time.monotonic()
        self._session = ort.InferenceSession(
            self._model_path, providers=providers
        )
        active = self._session.get_providers()
        self._input_name = self._session.get_inputs()[0].name
        elapsed = (time.monotonic() - t0) * 1000
        log.info(
            "HeadDetector loaded in %.0f ms  (providers: %s)",
            elapsed,
            active,
        )

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    @staticmethod
    def _preprocess(frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        """Resize + pad + normalise for YOLO input.

        Returns ``(blob, scale, pad_x, pad_y)`` where *scale* and pads
        map output coordinates back to the original frame.
        """
        h, w = frame.shape[:2]
        scale = _INPUT_SIZE / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))

        # Pad to square 640×640
        pad_x = (_INPUT_SIZE - new_w) // 2
        pad_y = (_INPUT_SIZE - new_h) // 2
        padded = np.full((_INPUT_SIZE, _INPUT_SIZE, 3), 114, dtype=np.uint8)
        padded[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

        # HWC→CHW, BGR→RGB, normalise to [0,1], add batch dim
        blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = blob[np.newaxis]  # (1, 3, 640, 640)
        return blob, scale, pad_x, pad_y

    # ------------------------------------------------------------------
    # Postprocessing
    # ------------------------------------------------------------------
    @staticmethod
    def _postprocess(
        output: np.ndarray,
        scale: float,
        pad_x: int,
        pad_y: int,
        conf_threshold: float,
        iou_threshold: float,
        frame_w: int,
        frame_h: int,
    ) -> list[tuple[int, int, int, int, float]]:
        """Decode YOLO output → list of ``(x, y, w, h, conf)`` in original coords."""
        # output shape: (1, 5, 8400) → transpose to (8400, 5)
        preds = output[0].T  # (8400, 5)

        # Filter by confidence
        confs = preds[:, 4]
        mask = confs >= conf_threshold
        preds = preds[mask]
        if len(preds) == 0:
            return []

        # cx, cy, w, h → x1, y1, x2, y2 for NMS
        cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2
        scores = preds[:, 4]

        # OpenCV NMS expects (N, 4) float32 and (N,) float32
        boxes_nms = np.stack([x1, y1, bw, bh], axis=1).astype(np.float32)
        indices = cv2.dnn.NMSBoxes(
            boxes_nms.tolist(), scores.tolist(), conf_threshold, iou_threshold
        )
        if len(indices) == 0:
            return []
        indices = np.array(indices).flatten()

        results: list[tuple[int, int, int, int, float]] = []
        for i in indices:
            # Map from padded 640×640 → original frame
            ox = (cx[i] - pad_x) / scale
            oy = (cy[i] - pad_y) / scale
            ow = bw[i] / scale
            oh = bh[i] / scale

            rx = int(max(0, ox - ow / 2))
            ry = int(max(0, oy - oh / 2))
            rw = int(min(ow, frame_w - rx))
            rh = int(min(oh, frame_h - ry))

            if rw > 0 and rh > 0:
                results.append((rx, ry, rw, rh, float(scores[i])))

        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
    ) -> list[tuple[int, int, int, int, float]]:
        """Detect all heads in *frame*.

        Returns a list of ``(x, y, w, h, confidence)`` tuples in the
        original frame coordinate space.
        """
        self._ensure_loaded()
        assert self._session is not None

        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        iou = iou_threshold if iou_threshold is not None else self.iou_threshold

        h, w = frame.shape[:2]
        blob, scale, pad_x, pad_y = self._preprocess(frame)
        raw = self._session.run(None, {self._input_name: blob})[0]
        return self._postprocess(raw, scale, pad_x, pad_y, conf, iou, w, h)
