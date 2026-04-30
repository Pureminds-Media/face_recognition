"""NVDEC-accelerated RTSP capture with cv2.VideoCapture compatibility.

Replaces ``cv2.VideoCapture`` for RTSP streams when NVIDIA hardware decoding
is available, freeing CPU cores by routing H.264/H.265/VP9/AV1 decode onto
the GPU's NVDEC engines. Falls back to software decode on a per-camera
basis if hardware decode init fails.

Drop-in API: ``read() -> (ok, frame_bgr_ndarray)``, ``release()``,
``isOpened()``, ``set()`` / ``get()`` for the common props.

Usage::

    from hw_capture import open_capture
    cap = open_capture("rtsp://...")
    ok, frame = cap.read()
    cap.release()

The opener tries hardware decode first; if that fails for any reason it
falls back to ``cv2.VideoCapture`` so existing reconnect/error logic in
the caller still works unchanged.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Map of base FFmpeg codec name → CUVID hardware decoder. PyAV/FFmpeg
# auto-detects the codec from the RTSP stream so we just need to look
# up which hardware decoder to substitute.
_CUVID_MAP = {
    "h264": "h264_cuvid",
    "hevc": "hevc_cuvid",
    "h265": "hevc_cuvid",
    "mpeg2video": "mpeg2_cuvid",
    "mpeg4": "mpeg4_cuvid",
    "vp8": "vp8_cuvid",
    "vp9": "vp9_cuvid",
    "av1": "av1_cuvid",
    "mjpeg": "mjpeg_cuvid",
    "vc1": "vc1_cuvid",
}


def _cuvid_for(codec_name: str) -> Optional[str]:
    return _CUVID_MAP.get((codec_name or "").lower())


class HwRtspCapture:
    """RTSP capture using PyAV + NVDEC.

    A background thread continuously demuxes + decodes from the input
    stream and stashes only the *most recent* frame, so ``read()`` always
    returns fresh data and old frames don't pile up. This mirrors how the
    existing grid worker uses ``cv2.VideoCapture`` with
    ``CAP_PROP_BUFFERSIZE=1``.
    """

    def __init__(self, url: str, codec_hint: Optional[str] = None):
        # Lazy import so callers that never hit hardware decode don't pay
        # the PyAV import cost.
        import av
        from av.codec.hwaccel import HWAccel

        self._url = url
        self._opened = False
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._latest_t: float = 0.0
        self._width = 0
        self._height = 0
        self._fps = 0.0

        # rtsp_transport=tcp matches OPENCV_FFMPEG_CAPTURE_OPTIONS in app.py
        # and prevents UDP packet loss / NAT issues common on Wi-Fi cams.
        # stimeout caps socket waits so a dead camera fails cleanly.
        options = {
            "rtsp_transport": "tcp",
            "stimeout": "5000000",  # microseconds
            "fflags": "nobuffer",
            "flags": "low_delay",
        }

        # Pass an HWAccel object so PyAV/FFmpeg routes decode onto NVDEC.
        # ``allow_software_fallback=False`` would force CUDA-only; we let
        # it stay True so a codec the GPU doesn't support falls back to
        # software inside this same container instead of erroring out.
        hwaccel = HWAccel(device_type="cuda", allow_software_fallback=True)

        self._container = av.open(
            url, options=options, timeout=10.0, hwaccel=hwaccel,
        )
        try:
            stream = self._container.streams.video[0]
        except IndexError:
            self._container.close()
            raise RuntimeError(f"No video stream in {url}")

        # Sanity-check that hwaccel actually attached. If not, bail so
        # the caller falls back to cv2.VideoCapture.
        if not getattr(stream.codec_context, "is_hwaccel", False):
            base_codec = stream.codec_context.codec.name if stream.codec_context.codec else "?"
            self._container.close()
            raise RuntimeError(
                f"hwaccel did not attach for codec {base_codec!r} on {url}"
            )

        self._stream = stream
        self._width = int(stream.codec_context.width or 0)
        self._height = int(stream.codec_context.height or 0)
        try:
            avg = stream.average_rate
            self._fps = float(avg) if avg else 0.0
        except Exception:
            self._fps = 0.0

        self._opened = True
        self._thread = threading.Thread(
            target=self._reader_loop, daemon=True, name=f"nvdec-{url[-32:]}"
        )
        self._thread.start()

    # ---------- background reader ----------
    def _reader_loop(self):
        try:
            for frame in self._container.decode(self._stream):
                if self._stop.is_set():
                    return
                # NVDEC produces NV12 frames in GPU memory; PyAV
                # transparently downloads + reformats to BGR via
                # libswscale. Cheaper than a CPU H.264/H.265 decode.
                try:
                    ndarr = frame.to_ndarray(format="bgr24")
                except Exception:
                    continue
                with self._lock:
                    self._latest = ndarr
                    self._latest_t = time.monotonic()
                    if self._width == 0:
                        self._height, self._width = ndarr.shape[:2]
        except Exception as e:
            log.warning("HwRtspCapture reader error on %s: %s", self._url, e)
        finally:
            self._opened = False

    # ---------- cv2.VideoCapture-compatible API ----------
    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            frame = self._latest
            self._latest = None  # consume so we don't hand the same frame twice
        if frame is None:
            return False, None
        return True, frame

    def release(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        try:
            self._container.close()
        except Exception:
            pass
        self._opened = False

    # cv2 callers occasionally probe these; keep the API surface
    # compatible so we can swap in without touching call sites.
    def set(self, prop: int, value) -> bool:
        # We can't meaningfully honour CAP_PROP_BUFFERSIZE etc. — our
        # background thread already enforces a one-frame buffer.
        return True

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        if prop == cv2.CAP_PROP_FPS:
            return float(self._fps)
        return 0.0


# Module-level toggle — set USE_NVDEC=0 in env to force software decode
# everywhere (useful for A/B comparison or if hardware decode misbehaves).
def _nvdec_enabled() -> bool:
    return os.getenv("USE_NVDEC", "1").strip().lower() not in ("0", "false", "no", "off")


def open_capture(source) -> object:
    """Return a capture object for ``source``.

    For RTSP/RTP/HTTP-streamed sources, attempt NVDEC. On any failure,
    fall back to ``cv2.VideoCapture`` with the existing FFmpeg options
    inherited from ``OPENCV_FFMPEG_CAPTURE_OPTIONS``.
    For non-network sources (webcam index, file path), always use
    ``cv2.VideoCapture``.
    """
    src_str = str(source) if source is not None else ""
    is_network = any(
        src_str.lower().startswith(p)
        for p in ("rtsp://", "rtp://", "http://", "https://")
    )

    if is_network and _nvdec_enabled():
        try:
            cap = HwRtspCapture(src_str)
            log.info("opened %s via NVDEC", src_str)
            return cap
        except Exception as e:
            log.warning("NVDEC failed for %s, falling back to CPU: %s", src_str, e)

    return cv2.VideoCapture(source)
