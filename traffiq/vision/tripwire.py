"""
Virtual tripwire and directional line-crossing mathematics for TRAFFIQ.
Calculates stop-line crossings, tracks cumulative flow counts, and renders overlay lines.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import cv2
import numpy as np

from traffiq.vision.tracker import BoundedIDSet


def calculate_centroid(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """
    Calculate integer (cx, cy) centroid from bounding box [x1, y1, x2, y2].

    Args:
        bbox: Sequence of (x1, y1, x2, y2) coordinates.

    Returns:
        Tuple of integer (cx, cy) pixel coordinates.
    """
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return cx, cy


def check_line_crossing(
    prev_cy: int,
    curr_cy: int,
    line_y: int,
    direction: str = "down",
) -> bool:
    """
    Evaluate whether vertical movement between prev_cy and curr_cy crossed line_y
    in the designated direction.

    Args:
        prev_cy: Centroid y-coordinate in previous frame.
        curr_cy: Centroid y-coordinate in current frame.
        line_y: Horizontal tripwire y-coordinate.
        direction: Target crossing direction ('down' for top-to-bottom, 'up' for bottom-to-top).

    Returns:
        True if motion satisfies directional crossing condition.
    """
    p_cy = int(prev_cy)
    c_cy = int(curr_cy)
    l_y = int(line_y)

    if direction == "down":
        return p_cy < l_y <= c_cy
    elif direction == "up":
        return p_cy > l_y >= c_cy
    return False


@dataclass(frozen=True)
class TripwireCrossing:
    """Records an occurrence of a vehicle crossing the tripwire."""

    track_id: int
    timestamp: float
    centroid: Tuple[int, int]
    previous_centroid: Tuple[int, int]
    direction: str


class VirtualTripwire:
    """
    Virtual horizontal tripwire for traffic flow counting and stop-line monitoring.
    Maintains line position, detects directional crossings, and prevents duplicate counting.
    """

    def __init__(
        self,
        y_position: Optional[int] = None,
        tripwire_ratio: float = 0.55,
        frame_height: Optional[int] = None,
        direction: str = "down",
        max_history: int = 10000,
    ) -> None:
        """
        Initialize VirtualTripwire with position geometry and direction.

        Args:
            y_position: Explicit integer y-pixel line coordinate.
            tripwire_ratio: Ratio of frame height for line placement (default 0.55 = 55%).
            frame_height: Total frame height to compute y_position if y_position is None.
            direction: Valid crossing direction ('down' or 'up').
            max_history: Bounded capacity for deduplication ID cache.
        """
        self.tripwire_ratio = float(tripwire_ratio)
        self.direction = str(direction).lower()

        if y_position is not None:
            self.y_position = int(y_position)
        elif frame_height is not None:
            self.y_position = int(self.tripwire_ratio * frame_height)
        else:
            self.y_position = int(self.tripwire_ratio * 720)

        self.counted_ids = BoundedIDSet(max_size=max_history)
        self._track_positions: Dict[int, Tuple[int, int]] = {}
        self.flow_count = 0

    def set_frame_height(self, frame_height: int) -> None:
        """Update y_position based on actual frame height."""
        self.y_position = int(self.tripwire_ratio * frame_height)

    def check_crossing(self, prev_cy: int, curr_cy: int) -> bool:
        """Check if motion crosses this tripwire's line in configured direction."""
        return check_line_crossing(prev_cy, curr_cy, self.y_position, self.direction)

    def update_track(self, track_id: int, centroid: Tuple[int, int]) -> bool:
        """
        Update vehicle position and evaluate crossing.
        If a new crossing occurs, records it and returns True.
        Suppresses duplicate crossing signals for already-counted vehicles.

        Args:
            track_id: Vehicle integer tracking ID.
            centroid: Current (cx, cy) pixel coordinates.

        Returns:
            True if vehicle crossed the line on this frame for the first time.
        """
        tid = int(track_id)
        cx = int(centroid[0])
        cy = int(centroid[1])

        if tid not in self._track_positions:
            self._track_positions[tid] = (cx, cy)
            return False

        _, prev_cy = self._track_positions[tid]
        self._track_positions[tid] = (cx, cy)

        if self.check_crossing(prev_cy, cy):
            if tid not in self.counted_ids:
                self.counted_ids.add(tid)
                self.flow_count += 1
                return True

        return False

    def draw(
        self,
        frame: np.ndarray,
        color: Optional[Tuple[int, int, int]] = None,
        label: Optional[str] = None,
    ) -> np.ndarray:
        """
        Draw visual tripwire stop-line and label on input frame.

        Args:
            frame: 3-channel BGR frame.
            color: Line color in BGR (default green).
            label: Text label (defaults to stop line and count).

        Returns:
            The modified frame with tripwire overlays.
        """
        h, w = frame.shape[:2]
        line_col = color if color is not None else (0, 255, 0)
        cv2.line(frame, (0, self.y_position), (w, self.y_position), line_col, 3)

        text = (
            label
            if label is not None
            else f"STOP LINE [y={self.y_position}] Passed: {self.flow_count}"
        )
        cv2.putText(
            frame,
            text,
            (10, max(20, self.y_position - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            line_col,
            2,
        )
        return frame


# Alias for compatibility with tests expecting Tripwire
Tripwire = VirtualTripwire
