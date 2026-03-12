"""Zero-shot action detection using CLIP (ONNX Runtime, GPU-accelerated).

Detects what a person is doing from a cropped image region using
OpenAI's CLIP model with configurable text labels.  No training needed —
just edit the ACTION_LABELS list to add or remove actions.

Usage
-----
    detector = ActionDetector()          # loads model on GPU
    label, confidence = detector.detect(frame, bbox, person_id="john")
    # label = "Using phone", confidence = 0.82
"""

from __future__ import annotations

import collections
import logging
import os
import time
from typing import Optional

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default action labels (office/desk environment).
#
# CLIP works best with prompts that describe visually distinct scenes.
# Each entry is (clip_prompt, display_name).
# ---------------------------------------------------------------------------
DEFAULT_ACTION_LABELS = [
    ("a photo of a person holding and looking at a cellphone in their hand",
     "Using phone"),
    ("a photo of a person reading a book or papers on a desk",
     "Reading"),
    ("a photo of a person with hands on a keyboard typing at a computer",
     "Typing"),
    ("a photo of a person writing on paper with a pen or pencil",
     "Writing"),
    ("a photo of a person holding a cup or bottle and drinking",
     "Drinking"),
    ("a photo of a person eating food or snacking",
     "Eating"),
    ("a photo of two people facing each other having a conversation",
     "Talking"),
    ("a photo of a person sitting quietly doing nothing with empty hands",
     "Idle"),
]

# Minimum bbox size (pixels) to attempt action detection.
# Matches the face detector's minimum (~20px) so action detection works at the same range.
_MIN_CROP_PX = 20

# Minimum crop dimensions fed to CLIP.  For distant/small faces the proportional
# padding would produce a tiny crop; these ensure enough visual context.
_MIN_CROP_W = 200
_MIN_CROP_H = 300

# Temporal smoothing: require this many consecutive same-label detections
# before switching the displayed action.
_SMOOTH_WINDOW = 5


class ActionDetector:
    """CLIP-based zero-shot action classifier.

    Parameters
    ----------
    model_path : str
        Local directory with the exported ONNX CLIP model, or a
        HuggingFace model ID (will be exported on first run).
    labels : list[tuple[str,str]] | None
        ``(clip_prompt, display_name)`` pairs.
        Defaults to ``DEFAULT_ACTION_LABELS``.
    confidence_threshold : float
        Minimum softmax probability to report an action.
        Below this, ``detect()`` returns ``(None, 0.0)``.
    smooth_window : int
        Number of consecutive same-label detections required before
        switching the displayed label.  Prevents rapid flickering.
    provider : str
        ONNX Runtime execution provider.
    """

    def __init__(
        self,
        model_path: str = "models/clip-vit-base-patch32-onnx",
        labels: list[tuple[str, str]] | None = None,
        confidence_threshold: float = 0.35,
        smooth_window: int = _SMOOTH_WINDOW,
        provider: str = "CUDAExecutionProvider",
    ):
        pairs = labels or list(DEFAULT_ACTION_LABELS)
        self._prompts = [p for p, _ in pairs]
        self._display_names = [d for _, d in pairs]
        self.confidence_threshold = confidence_threshold
        self._smooth_window = smooth_window
        self._model = None
        self._processor = None
        self._provider = provider
        self._model_path = model_path
        self._ready = False

        # Per-person smoothing history:  person_id -> deque of display_names
        self._history: dict[str, collections.deque] = {}
        # Per-person current stable label
        self._stable: dict[str, tuple[str, float]] = {}  # person_id -> (label, conf)

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------
    def _ensure_loaded(self):
        if self._ready:
            return
        try:
            from optimum.onnxruntime import ORTModelForZeroShotImageClassification
            from transformers import CLIPProcessor
            import onnxruntime as ort

            # Suppress ONNX Memcpy/provider assignment warnings
            sess_opts = ort.SessionOptions()
            sess_opts.log_severity_level = 3

            t0 = time.monotonic()
            if os.path.isdir(self._model_path):
                log.info("Loading CLIP ONNX model from %s", self._model_path)
                self._model = ORTModelForZeroShotImageClassification.from_pretrained(
                    self._model_path, provider=self._provider,
                    session_options=sess_opts,
                )
            else:
                log.info("Exporting CLIP to ONNX from %s", self._model_path)
                self._model = ORTModelForZeroShotImageClassification.from_pretrained(
                    "openai/clip-vit-base-patch32",
                    export=True,
                    provider=self._provider,
                    session_options=sess_opts,
                )
                os.makedirs(self._model_path, exist_ok=True)
                self._model.save_pretrained(self._model_path)

            self._model.use_io_binding = False
            self._processor = CLIPProcessor.from_pretrained(
                self._model_path, use_fast=False
            )
            elapsed = time.monotonic() - t0
            log.info("CLIP action detector ready (%.1fs)", elapsed)
            self._ready = True
        except Exception:
            log.exception("Failed to load CLIP action detector")
            self._ready = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        person_id: str = "",
    ) -> tuple[Optional[str], float]:
        """Classify the action of a person in *bbox*.

        Parameters
        ----------
        frame : np.ndarray
            Full camera frame (BGR, HxWx3).
        bbox : (x, y, w, h)
            Bounding box of the person in the frame.
        person_id : str
            Identifier for temporal smoothing (e.g. person name).
            If empty, smoothing is skipped.

        Returns
        -------
        (display_label, confidence) or (None, 0.0) if below threshold.
        """
        self._ensure_loaded()
        if not self._ready or self._model is None:
            return (None, 0.0)

        x, y, w, h = [int(v) for v in bbox]
        if w < _MIN_CROP_PX or h < _MIN_CROP_PX:
            return (None, 0.0)

        H, W = frame.shape[:2]

        # Expand crop to include torso/hands (important for action cues)
        # More padding below and to sides, less above
        pad_x = int(w * 0.5)
        pad_y_top = int(h * 0.2)
        pad_y_bot = int(h * 1.0)  # include torso/hands below face

        # For distant/small faces, ensure the crop meets a minimum size
        # so CLIP has enough visual context to classify action.
        crop_w = w + 2 * pad_x
        crop_h = h + pad_y_top + pad_y_bot
        if crop_w < _MIN_CROP_W:
            pad_x = (_MIN_CROP_W - w) // 2
        if crop_h < _MIN_CROP_H:
            extra = _MIN_CROP_H - crop_h
            pad_y_bot += extra  # extend downward (torso/hands more useful)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y_top)
        x2 = min(W, x + w + pad_x)
        y2 = min(H, y + h + pad_y_bot)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return (None, 0.0)

        # Convert BGR → RGB → PIL
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        try:
            inputs = self._processor(
                text=self._prompts,
                images=pil_img,
                return_tensors="np",
                padding=True,
            )
            outputs = self._model(**inputs)
            logits = np.asarray(outputs.logits_per_image[0], dtype=np.float32)

            # Softmax
            exp = np.exp(logits - np.max(logits))
            probs = exp / exp.sum()

            best_idx = int(np.argmax(probs))
            best_prob = float(probs[best_idx])
            raw_label = self._display_names[best_idx]

            if best_prob < self.confidence_threshold:
                return self._get_stable(person_id)

            # Temporal smoothing
            if person_id and self._smooth_window > 1:
                return self._smooth(person_id, raw_label, best_prob)

            return (raw_label, best_prob)
        except Exception:
            log.exception("Action detection failed")
            return (None, 0.0)

    def _smooth(
        self, person_id: str, raw_label: str, conf: float
    ) -> tuple[Optional[str], float]:
        """Apply temporal smoothing: only switch label after N consecutive
        same-label detections."""
        if person_id not in self._history:
            self._history[person_id] = collections.deque(maxlen=self._smooth_window)

        hist = self._history[person_id]
        hist.append(raw_label)

        # Check if the last N entries are all the same
        if len(hist) >= self._smooth_window and len(set(hist)) == 1:
            self._stable[person_id] = (raw_label, conf)
            return (raw_label, conf)

        # Otherwise return the current stable label (or None if no stable yet)
        return self._get_stable(person_id)

    def _get_stable(self, person_id: str) -> tuple[Optional[str], float]:
        """Return the current stable label for a person, or (None, 0.0)."""
        if person_id:
            return self._stable.get(person_id, (None, 0.0))
        return (None, 0.0)

    def clear_person(self, person_id: str):
        """Clear smoothing history for a person (e.g. when they leave)."""
        self._history.pop(person_id, None)
        self._stable.pop(person_id, None)

    @property
    def is_ready(self) -> bool:
        return self._ready

    def set_labels(self, labels: list[tuple[str, str]]):
        """Update action labels at runtime."""
        self._prompts = [p for p, _ in labels]
        self._display_names = [d for _, d in labels]
        # Clear all smoothing state since labels changed
        self._history.clear()
        self._stable.clear()
