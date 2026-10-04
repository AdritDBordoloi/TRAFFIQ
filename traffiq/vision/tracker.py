"""
Vehicle tracking module wrapping ByteTrack for TRAFFIQ.
Implements bounded frame-based TTL trajectory management to prevent memory leaks
without destroying active in-flight trajectories.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np

from traffiq.vision.detector import COCO_VEHICLE_CLASSES, Detection, resolve_device


class BoundedIDSet:
    """
    A bounded set implementation using an internal deque to guarantee
    fixed upper bounds on memory while preserving fast O(1) membership checks.
    """

    def __init__(self, max_size: int = 10000) -> None:
        self.max_size = int(max_size)
        self._set: Set[int] = set()
        self._queue: deque[int] = deque()

    def add(self, item: int) -> None:
        """Add an item to the bounded set, evicting oldest entry if full."""
        int_item = int(item)
        if int_item not in self._set:
            if len(self._queue) >= self.max_size:
                evicted = self._queue.popleft()
                self._set.discard(evicted)
            self._queue.append(int_item)
            self._set.add(int_item)

    def __contains__(self, item: object) -> bool:
        try:
            return int(item) in self._set  # type: ignore
        except (ValueError, TypeError):
            return False

    def __len__(self) -> int:
        return len(self._set)

    def __iter__(self):
        return iter(self._set)


class TrajectoryCache:
    """
    Bounded trajectory cache using frame-based TTL (Time-To-Live) eviction.
    Evicts only tracks unseen for more than TTL frames, never wiping active trajectories.
    """

    def __init__(self, ttl: int = 30) -> None:
        self.ttl = int(ttl)
        # Internal storage: {track_id: {"last_seen": frame_idx, "positions": [(cx, cy), ...]}}
        self._cache: Dict[int, Dict[str, Any]] = {}

    def update(self, track_id: int, centroid: Tuple[int, int], frame_idx: int) -> None:
        """
        Record vehicle centroid position and update its last seen frame timestamp.

        Args:
            track_id: Unique integer vehicle tracking identifier.
            centroid: (cx, cy) integer pixel coordinates.
            frame_idx: Monotonic sequence index of the current video frame.
        """
        tid = int(track_id)
        pt = (int(centroid[0]), int(centroid[1]))

        if tid not in self._cache:
            self._cache[tid] = {
                "last_seen": int(frame_idx),
                "positions": [],
            }

        self._cache[tid]["last_seen"] = int(frame_idx)
        self._cache[tid]["positions"].append(pt)

    def prune_inactive(self, current_frame: int, ttl: Optional[int] = None) -> int:
        """
        Evict track IDs unseen for more than TTL frames.
        Never touches active tracks whose last_seen frame is within TTL.

        Args:
            current_frame: Current video frame index.
            ttl: Optional override for the eviction threshold.

        Returns:
            Number of evicted track records.
        """
        effective_ttl = self.ttl if ttl is None else int(ttl)
        curr = int(current_frame)

        to_evict = [
            tid
            for tid, data in self._cache.items()
            if (curr - data["last_seen"]) > effective_ttl
        ]

        for tid in to_evict:
            del self._cache[tid]

        return len(to_evict)

    def get_trajectory(self, track_id: int) -> Optional[List[Tuple[int, int]]]:
        """Return full history of positions for track_id, or None if not present."""
        tid = int(track_id)
        if tid in self._cache:
            return list(self._cache[tid]["positions"])
        return None

    def get_last_position(self, track_id: int) -> Optional[Tuple[int, int]]:
        """Return the most recent centroid coordinate for track_id."""
        tid = int(track_id)
        if tid in self._cache and self._cache[tid]["positions"]:
            return self._cache[tid]["positions"][-1]
        return None

    def active_track_count(self) -> int:
        """Return count of currently maintained tracking trajectories."""
        return len(self._cache)

    def clear(self) -> None:
        """Explicit cache clear (used only for full reset)."""
        self._cache.clear()


@dataclass(frozen=True)
class TrackedVehicle:
    """Represents an active vehicle tracked across frames by ByteTrack."""

    track_id: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    class_id: int
    class_name: str
    centroid: Tuple[int, int]  # (cx, cy)
    confidence: float = 0.0

    @property
    def width(self) -> int:
        return max(0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        return max(0, self.bbox[3] - self.bbox[1])


class VehicleTracker:
    """
    ByteTrack-based multi-object tracker for vehicular traffic.
    Maintains persistent trajectory history with safe frame-based TTL pruning
    and bounded deduplication sets.
    """

    def __init__(
        self,
        model_path: str = "yolov8s.pt",
        tracker_config: str = "bytetrack.yaml",
        confidence_threshold: float = 0.25,
        device: Union[int, str] = "0",
        classes: Optional[Union[Set[int], Sequence[int]]] = None,
        ttl: int = 30,
        max_history: int = 10000,
        model: Optional[object] = None,
    ) -> None:
        """
        Initialize VehicleTracker with model, tracker parameters, and memory caches.

        Args:
            model_path: Path to YOLO model weights.
            tracker_config: ByteTrack configuration filename (e.g. 'bytetrack.yaml').
            confidence_threshold: Detection confidence cutoff.
            device: Inference device ('0', 'cpu').
            classes: COCO class IDs to track (defaults to standard vehicle classes).
            ttl: Frames of absence before an inactive track is pruned.
            max_history: Upper bound for tracked ID deduplication memory.
            model: Optional pre-loaded YOLO model.
        """
        self.model_path = str(model_path)
        self.tracker_config = str(tracker_config)
        self.confidence_threshold = float(confidence_threshold)
        self.device = resolve_device(device)
        self.ttl = int(ttl)

        if classes is not None:
            self.classes: Set[int] = set(classes)
        else:
            self.classes = set(COCO_VEHICLE_CLASSES.keys())

        self.class_names: Dict[int, str] = {
            cid: COCO_VEHICLE_CLASSES.get(cid, "Vehicle") for cid in self.classes
        }

        self._model = model
        self.trajectory_cache = TrajectoryCache(ttl=self.ttl)
        self.counted_ids = BoundedIDSet(max_size=max_history)
        self.violated_ids = BoundedIDSet(max_size=max_history)
        self._current_frame_idx = 0

    @property
    def model(self) -> object:
        """Lazy loader for YOLO model instance."""
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
        return self._model

    def track(self, frame: np.ndarray, frame_idx: Optional[int] = None) -> List[TrackedVehicle]:
        """
        Run ByteTrack tracking on an input video frame.

        Args:
            frame: 3-channel BGR numpy array.
            frame_idx: Optional frame index; increments internally if omitted.

        Returns:
            List of TrackedVehicle instances with native integer track IDs.
        """
        if frame is None or frame.size == 0:
            return []

        if frame_idx is not None:
            self._current_frame_idx = int(frame_idx)
        else:
            self._current_frame_idx += 1

        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_config,
            classes=list(self.classes),
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )[0]

        tracked_vehicles: List[TrackedVehicle] = []
        if (
            results.boxes is not None
            and results.boxes.id is not None
            and len(results.boxes) > 0
        ):
            boxes = results.boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            confs = (
                results.boxes.conf.cpu().numpy().astype(float)
                if results.boxes.conf is not None
                else [0.0] * len(boxes)
            )

            for box, raw_tid, cls_id, conf in zip(boxes, track_ids, class_ids, confs):
                tid = int(raw_tid)  # Guarantee native Python int
                cid = int(cls_id)
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                cname = self.class_names.get(cid, "Vehicle")

                # Update trajectory history
                self.trajectory_cache.update(
                    track_id=tid,
                    centroid=(cx, cy),
                    frame_idx=self._current_frame_idx,
                )

                tracked_vehicles.append(
                    TrackedVehicle(
                        track_id=tid,
                        bbox=(x1, y1, x2, y2),
                        class_id=cid,
                        class_name=cname,
                        centroid=(cx, cy),
                        confidence=float(conf),
                    )
                )

        return tracked_vehicles

    def update(
        self,
        frame_or_detections: Any,
        frame_or_idx: Any = None,
        frame_idx: Optional[int] = None,
    ) -> List[TrackedVehicle]:
        """
        Polymorphic update method conforming to common tracker interfaces.
        Accepts (frame, frame_idx) or (detections, frame).
        """
        if isinstance(frame_or_detections, np.ndarray):
            f_idx = frame_or_idx if isinstance(frame_or_idx, int) else frame_idx
            return self.track(frame_or_detections, frame_idx=f_idx)

        # If detections passed as first arg and frame as second
        if isinstance(frame_or_idx, np.ndarray):
            return self.track(frame_or_idx, frame_idx=frame_idx)

        return []

    def prune_inactive(self, current_frame: int, ttl: Optional[int] = None) -> int:
        """
        Evict trajectories unseen for more than TTL frames.

        Args:
            current_frame: Current video frame index.
            ttl: Optional override TTL frames count.

        Returns:
            Number of evicted trajectories.
        """
        return self.trajectory_cache.prune_inactive(current_frame, ttl=ttl)

    def get_trajectory(self, track_id: int) -> Optional[List[Tuple[int, int]]]:
        """Return trajectory points for a given track ID."""
        return self.trajectory_cache.get_trajectory(track_id)
