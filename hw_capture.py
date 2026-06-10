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


class _NvdecRWLock:
    """Readers-writer lock for NVDEC context operations.

    Readers  — frame.to_ndarray() calls (many may run concurrently).
    Writers  — av.open() and container.close() (exclusive; waits for all
               active readers to finish before proceeding).

    Writers get priority: once a write is queued, new readers block so the
    writer is not starved by a steady stream of decode transfers.
    """

    def __init__(self):
        self._cv = threading.Condition(threading.Lock())
        self._readers: int = 0
        self._writers_waiting: int = 0
        self._writing: bool = False

    def acquire_read(self, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        with self._cv:
            while self._writing or self._writers_waiting > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cv.wait(timeout=min(remaining, 0.1))
            self._readers += 1
            return True

    def release_read(self) -> None:
        with self._cv:
            self._readers -= 1
            if self._readers == 0:
                self._cv.notify_all()

    def acquire_write(self, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        with self._cv:
            self._writers_waiting += 1
            while self._readers > 0 or self._writing:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._writers_waiting -= 1
                    return False
                self._cv.wait(timeout=min(remaining, 0.1))
            self._writers_waiting -= 1
            self._writing = True
            return True

    def release_write(self) -> None:
        with self._cv:
            self._writing = False
            self._cv.notify_all()


# Global RW lock for all NVDEC operations.
# to_ndarray() holds a read slot; av.open() / container.close() hold the
# write slot (exclusive).  This prevents container teardown from racing with
# active GPU→CPU transfers on other cameras, which causes SIGSEGV in the
# NVDEC driver.
_nvdec_rw = _NvdecRWLock()

# Legacy simple lock kept for the _nvdec_reconnect_sem path in face_engine.py
# which imports it by name.  The RW lock supersedes it for actual CUDA ops.
_nvdec_lock = threading.Lock()

# Limit concurrent camera reconnects.  Each reconnect does: close old context
# (joins its reader thread, up to ~7 s) then opens a new context.
# NVDEC_RECONNECT_CONCURRENCY: how many cameras may reconnect simultaneously.
#   Default 1 — safest; one reconnect at a time prevents driver state churn.
#   The RW lock now handles the deeper race, but keeping concurrency low
#   avoids a thundering-herd of simultaneous av.open() calls.
def _nvdec_reconnect_concurrency() -> int:
    try:
        return max(1, int(os.getenv("NVDEC_RECONNECT_CONCURRENCY", "1")))
    except (ValueError, TypeError):
        return 1

_nvdec_reconnect_sem = threading.Semaphore(_nvdec_reconnect_concurrency())
# Monotonic time of the last reconnect completion; enforces a minimum gap
# between consecutive reconnect operations to avoid back-to-back NVDEC churn.
_nvdec_last_reconnect_t: float = 0.0
# NVDEC_RECONNECT_GAP_SECS: minimum seconds between consecutive NVDEC
# open/close cycles.
_NVDEC_RECONNECT_GAP = float(os.getenv("NVDEC_RECONNECT_GAP_SECS", "2.0"))

# Limit concurrent GPU→CPU frame downloads (frame.to_ndarray on NVDEC frames).
# MAX_NVDEC_TRANSFERS: default 8 — suitable for RTX 5090 (2 NVDEC engines).
# Lower to 3-4 for laptop/mobile GPUs (RTX 40 series and below).
def _max_nvdec_transfers() -> int:
    try:
        return max(1, int(os.getenv("MAX_NVDEC_TRANSFERS", "8")))
    except (ValueError, TypeError):
        return 8

_nvdec_transfer_sem: Optional[threading.Semaphore] = None
_nvdec_transfer_sem_lock = threading.Lock()

def _get_transfer_sem() -> threading.Semaphore:
    global _nvdec_transfer_sem
    with _nvdec_transfer_sem_lock:
        if _nvdec_transfer_sem is None:
            _nvdec_transfer_sem = threading.Semaphore(_max_nvdec_transfers())
    return _nvdec_transfer_sem

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
        _stimeout = os.getenv("RTSP_CONNECT_TIMEOUT_MS", "2000")
        options = {
            "rtsp_transport": "tcp",
            "stimeout": str(int(_stimeout) * 1000),  # microseconds
            "fflags": "nobuffer",
            "flags": "low_delay",
        }

        # Pass an HWAccel object so PyAV/FFmpeg routes decode onto NVDEC.
        # allow_software_fallback=False: unsupported codecs raise an exception
        # which is caught by open_capture() and falls back to cv2 instead.
        # Keeping it False avoids PyAV allocating both GPU and CPU frame
        # buffers simultaneously, which roughly doubles VRAM consumption.
        hwaccel = HWAccel(device_type="cuda", allow_software_fallback=False)

        _av_timeout = int(_stimeout) / 1000.0 + 0.5  # slightly above stimeout

        # av.open() is a write operation — must not run while any other
        # camera's to_ndarray() is in progress (NVDEC driver shared state).
        if not _nvdec_rw.acquire_write(timeout=30.0):
            raise RuntimeError(f"Timed out waiting for NVDEC write lock: {url}")
        try:
            self._container = av.open(
                url, options=options, timeout=_av_timeout, hwaccel=hwaccel,
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
        finally:
            _nvdec_rw.release_write()

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
            # Split demux (network I/O) from decode (NVDEC hardware) so the
            # read lock is only held during actual GPU operations, not while
            # waiting for the next packet from the network.
            #
            # Without this split, a stalled camera would hold the read lock
            # indefinitely, blocking all reconnect writes on other cameras.
            #
            # Timeline per packet:
            #   container.demux() → reads encoded packet from network  [no lock]
            #   packet.decode()   → NVDEC hardware decode              [read lock]
            #   frame.to_ndarray()→ GPU→CPU transfer                   [read lock]
            for packet in self._container.demux(self._stream):
                if self._stop.is_set():
                    break
                # flush packet — end of stream sentinel from PyAV
                if packet.dts is None:
                    break
                # Encoded packet is now in CPU memory; no NVDEC touched yet.
                # Acquire read slot only for the hardware decode + transfer.
                if not _nvdec_rw.acquire_read(timeout=2.0):
                    continue  # reconnect write pending; drop this packet
                try:
                    for frame in packet.decode():
                        with _get_transfer_sem():
                            ndarr = frame.to_ndarray(format="bgr24")
                        with self._lock:
                            self._latest = ndarr
                            self._latest_t = time.monotonic()
                            if self._width == 0:
                                self._height, self._width = ndarr.shape[:2]
                except Exception:
                    pass
                finally:
                    _nvdec_rw.release_read()
        except Exception as e:
            log.warning("HwRtspCapture reader error on %s: %s", self._url, e)
        finally:
            # container.close() is a write operation — waits for all active
            # decode+transfer ops on other cameras to complete first.
            if not _nvdec_rw.acquire_write(timeout=30.0):
                log.warning("HwRtspCapture: timed out waiting for write lock on close: %s", self._url)
            try:
                self._container.close()
            except Exception:
                pass
            finally:
                _nvdec_rw.release_write()
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
        # The reader thread owns container.close() — it calls it in its finally
        # block after decode() has returned.  We just need to wait for the
        # reader to finish.  stimeout=2 s means a stuck network read will abort
        # within ~2 s of _stop being set; 7 s gives ample margin.
        if self._thread is not None:
            self._thread.join(timeout=7.0)
            if self._thread.is_alive():
                log.warning("HwRtspCapture: reader thread did not exit in 7 s for %s", self._url)
            self._thread = None
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
            _is_network_err = any(
                kw in str(e).lower()
                for kw in ("no route to host", "connection refused", "timed out",
                           "name or service not known", "immediate exit")
            )
            if _is_network_err:
                log.warning("Camera unreachable (network error) %s: %s", src_str, e)
            else:
                log.warning("NVDEC failed for %s, falling back to CPU: %s", src_str, e)

    return cv2.VideoCapture(source)
