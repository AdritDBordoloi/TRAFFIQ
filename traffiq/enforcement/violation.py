"""
Violation detection and red light infraction evaluator for TRAFFIQ.
Monitors vehicle tripwire crossings during RED signal state, extracts vehicle evidence crops,
and creates structured ViolationEvent instances.
"""

from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np

from traffiq.enforcement.signal import SignalState
from traffiq.vision.tracker import BoundedIDSet


@dataclass
class ViolationEvent:
    """Represents an infractions event where a vehicle crossed the stop-line on RED."""

    track_id: int
    timestamp: str
    signal_state: str
    bbox: Tuple[int, int, int, int]
    vehicle_crop: np.ndarray
    plate_number: str = "UNKNOWN"
    confidence: float = 0.0
    fine_amount: int = 1000
    violation_type: str = "Red Light Jump"
    evidence_path: str = ""

    def __post_init__(self) -> None:
        self.track_id = int(self.track_id)
        self.fine_amount = int(self.fine_amount)


class ViolationDetector:
    """
    Evaluates stop-line infractions against active traffic signal status.
    Generates violation events when tracked vehicles cross the tripwire during RED lights.
    """

    def __init__(
        self,
        fine_amount: int = 1000,
        violation_type: str = "Red Light Jump",
        max_history: int = 10000,
    ) -> None:
        """
        Initialize ViolationDetector.

        Args:
            fine_amount: Default fine penalty amount in local currency (e.g. INR 1000).
            violation_type: Violation description label.
            max_history: Capacity for bounded deduplication of violated track IDs.
        """
        self.fine_amount = int(fine_amount)
        self.violation_type = str(violation_type)
        self.violated_ids = BoundedIDSet(max_size=max_history)

    def crop_vehicle(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        padding: int = 10,
    ) -> np.ndarray:
        """
        Crop vehicle bounding box from frame with border safety padding.

        Args:
            frame: 3-channel BGR source frame.
            bbox: (x1, y1, x2, y2) coordinates.
            padding: Additional perimeter padding pixels.

        Returns:
            Extracted vehicle image slice.
        """
        if frame is None or frame.size == 0:
            return np.empty((0, 0, 3), dtype=np.uint8)

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        crop_y1 = max(0, y1 - padding)
        crop_y2 = min(h, y2 + padding)
        crop_x1 = max(0, x1 - padding)
        crop_x2 = min(w, x2 + padding)

        return frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()

    def evaluate_crossing(
        self,
        track_id: int,
        crossed: bool,
        signal_state: Union[SignalState, str],
        bbox: Tuple[int, int, int, int],
        frame: np.ndarray,
        timestamp: Optional[str] = None,
    ) -> Optional[ViolationEvent]:
        """
        Evaluate whether a vehicle crossing constitutes a red light violation.

        Args:
            track_id: Integer tracking ID.
            crossed: True if the vehicle crossed the stop-line tripwire on this frame.
            signal_state: Current traffic signal state ('GREEN' or 'RED').
            bbox: Vehicle bounding box (x1, y1, x2, y2).
            frame: Full video frame image array.
            timestamp: Optional formatted timestamp string.

        Returns:
            ViolationEvent if an infraction is confirmed, otherwise None.
        """
        tid = int(track_id)
        if isinstance(signal_state, SignalState):
            s_val = signal_state.value
        else:
            s_val = str(signal_state).strip().upper()

        if crossed and s_val == "RED":
            if tid not in self.violated_ids:
                self.violated_ids.add(tid)
                ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                crop = self.crop_vehicle(frame, bbox)

                return ViolationEvent(
                    track_id=tid,
                    timestamp=ts,
                    signal_state=s_val,
                    bbox=bbox,
                    vehicle_crop=crop,
                    fine_amount=self.fine_amount,
                    violation_type=self.violation_type,
                )

        return None

    def is_violation(self, track_id: int) -> bool:
        """Return True if vehicle ID has been marked as a violator."""
        return int(track_id) in self.violated_ids

    def get_violation_count(self) -> int:
        """Return total count of recorded violations."""
        return len(self.violated_ids)
