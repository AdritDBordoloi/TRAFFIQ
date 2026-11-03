"""
Safe camera stream and video capture management for TRAFFIQ.
Provides context-managed acquisition and deterministic resource release.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Generator, Optional, Tuple, Union
import cv2
import numpy as np


class CameraStream:
    """
    Context-managed camera stream wrapper around cv2.VideoCapture.
    Guarantees resource release (cap.release() and cv2.destroyAllWindows()) in finally blocks,
    handles Windows DirectShow (cv2.CAP_DSHOW), custom resolution configuration,
    hardware buffer size tuning, optional threaded acquisition, and safe frame reading.
    """

    def __init__(
        self,
        source: Union[int, str] = 0,
        width: int = 1280,
        height: int = 720,
        api_preference: Optional[int] = None,
        auto_open: bool = True,
        threaded: bool = False,
    ) -> None:
        """
        Initialize CameraStream with capture source and target resolution.

        Args:
            source: Camera device index (int) or video file / RTSP path (str).
            width: Target frame width in pixels (default 1280).
            height: Target frame height in pixels (default 720).
            api_preference: Preferred OpenCV VideoCapture API backend (e.g. cv2.CAP_DSHOW).
            auto_open: Whether to automatically open the capture upon instantiation.
        """
        # Convert numeric string to int if applicable
        if isinstance(source, str) and source.strip().isdigit():
            self.source: Union[int, str] = int(source.strip())
        else:
            self.source = source

        self.width = int(width)
        self.height = int(height)
        self.api_preference = api_preference
        self.threaded = bool(threaded)
        self.cap: Optional[cv2.VideoCapture] = None

        self._thread: Optional[threading.Thread] = None
        self._thread_running = False
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()

        # Determine default API backend on Windows
        if self.api_preference is None and isinstance(self.source, int):
            if sys.platform.startswith("win"):
                self.api_preference = getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY)
            else:
                self.api_preference = cv2.CAP_ANY

        if auto_open:
            self.open()

    def open(self) -> cv2.VideoCapture:
        """
        Open the underlying VideoCapture device and apply hardware properties.

        Returns:
            The opened cv2.VideoCapture instance.
        """
        if self.cap is not None and self.cap.isOpened():
            return self.cap

        try:
            if self.api_preference is not None:
                self.cap = cv2.VideoCapture(self.source, self.api_preference)
            else:
                self.cap = cv2.VideoCapture(self.source)

            if self.cap.isOpened():
                # Hardware buffer size 1 to minimize latency and frame queue overhead
                try:
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                # On Windows with live camera, request MJPG fourcc for high USB throughput
                if sys.platform.startswith("win") and isinstance(self.source, int):
                    try:
                        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                    except Exception:
                        pass
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))

                if self.threaded:
                    self._start_thread()
        except Exception:
            self.release()
            raise

        return self.cap

    def is_opened(self) -> bool:
        """Return True if the video capture stream is currently open."""
        return bool(self.cap is not None and self.cap.isOpened())

    def _start_thread(self) -> None:
        """Start background daemon thread for continuous frame reading."""
        if not self._thread_running and self.cap is not None and self.cap.isOpened():
            self._thread_running = True
            self._thread = threading.Thread(target=self._reader_loop, daemon=True, name="CameraStreamReader")
            self._thread.start()

    def _reader_loop(self) -> None:
        """Dedicated background reader loop updating the latest frame buffer."""
        while self._thread_running and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self._frame_lock:
                    self._latest_frame = frame
            elif isinstance(self.source, str) and not str(self.source).isdigit():
                try:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                except Exception:
                    pass
                time.sleep(0.005)
            else:
                time.sleep(0.005)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame from the stream.

        Returns:
            Tuple of (success_boolean, frame_array_or_None).
        """
        if self.threaded:
            if self._latest_frame is None and self._thread_running:
                for _ in range(50):
                    with self._frame_lock:
                        if self._latest_frame is not None:
                            break
                    time.sleep(0.005)

            with self._frame_lock:
                if self._latest_frame is not None:
                    return True, self._latest_frame
            return False, None

        if self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None
        return True, frame

    def release(self) -> None:
        """
        Deterministically release video capture handle and destroy all GUI windows.
        Always executed inside a try...finally block.
        """
        self._thread_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

        try:
            if self.cap is not None:
                self.cap.release()
        finally:
            self.cap = None
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    def __enter__(self) -> CameraStream:
        """Enter context manager, opening capture if needed."""
        if not self.is_opened():
            self.open()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exit context manager, guaranteeing hardware handle release."""
        self.release()

    def __iter__(self) -> Generator[np.ndarray, None, None]:
        """Iterate over frames until stream ends or is closed."""
        while self.is_opened():
            ret, frame = self.read()
            if not ret or frame is None:
                break
            yield frame

    def frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Yield indexed (frame_idx, frame) pairs."""
        idx = 0
        while self.is_opened():
            ret, frame = self.read()
            if not ret or frame is None:
                break
            yield idx, frame
            idx += 1

    @property
    def fps(self) -> float:
        """Return frame rate of the capture device if queryable."""
        if self.cap is not None and self.cap.isOpened():
            val = self.cap.get(cv2.CAP_PROP_FPS)
            return float(val) if val > 0 else 30.0
        return 30.0

    @property
    def actual_size(self) -> Tuple[int, int]:
        """Return actual (width, height) of the opened stream."""
        if self.cap is not None and self.cap.isOpened():
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if w > 0 and h > 0:
                return w, h
        return self.width, self.height
