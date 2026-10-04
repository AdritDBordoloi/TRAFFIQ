"""
Safe camera stream and video capture management for TRAFFIQ.
Provides context-managed acquisition and deterministic resource release.
"""

from __future__ import annotations

import sys
from typing import Generator, Optional, Tuple, Union
import cv2
import numpy as np


class CameraStream:
    """
    Context-managed camera stream wrapper around cv2.VideoCapture.
    Guarantees resource release (cap.release() and cv2.destroyAllWindows()) in finally blocks,
    handles Windows DirectShow (cv2.CAP_DSHOW), custom resolution configuration,
    and safe frame reading.
    """

    def __init__(
        self,
        source: Union[int, str] = 0,
        width: int = 1280,
        height: int = 720,
        api_preference: Optional[int] = None,
        auto_open: bool = True,
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
        self.cap: Optional[cv2.VideoCapture] = None

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
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        except Exception:
            self.release()
            raise

        return self.cap

    def is_opened(self) -> bool:
        """Return True if the video capture stream is currently open."""
        return bool(self.cap is not None and self.cap.isOpened())

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame from the stream.

        Returns:
            Tuple of (success_boolean, frame_array_or_None).
        """
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
