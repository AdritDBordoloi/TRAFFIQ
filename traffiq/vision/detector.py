"""
Vehicle detection module using YOLOv8 for the TRAFFIQ system.
Filters detections to COCO vehicular classes with device fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np

# Standard COCO vehicle category mappings
COCO_VEHICLE_CLASSES: Dict[int, str] = {
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}


def resolve_device(device: Union[int, str] = "0") -> Union[int, str]:
    """
    Resolve inference device with automatic fallback to CPU if CUDA is unavailable.

    Args:
        device: Device request ('0', 'cpu', 'cuda', etc.).

    Returns:
        Integer device ID for CUDA GPU, or 'cpu' string.
    """
    dev_str = str(device).strip().lower()
    if dev_str in ("cpu", "-1"):
        return "cpu"

    try:
        import torch
        if torch.cuda.is_available():
            if dev_str.isdigit():
                return int(dev_str)
            return dev_str
        return "cpu"
    except ImportError:
        return "cpu"


@dataclass(frozen=True)
class Detection:
    """Represents a single detected vehicle instance in a frame."""

    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixel coordinates
    confidence: float
    class_id: int
    class_name: str

    @property
    def centroid(self) -> Tuple[int, int]:
        """Compute integer (cx, cy) centroid from bounding box."""
        x1, y1, x2, y2 = self.bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    @property
    def width(self) -> int:
        return max(0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> int:
        return max(0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> int:
        return self.width * self.height


class VehicleDetector:
    """
    YOLOv8 vehicle detector filtered to road transport classes:
    bicycle (1), car (2), motorcycle (3), bus (5), truck (7).
    """

    def __init__(
        self,
        model_path: str = "yolov8s.pt",
        confidence_threshold: float = 0.25,
        device: Union[int, str] = "0",
        classes: Optional[Union[Set[int], Sequence[int]]] = None,
        imgsz: int = 640,
        model: Optional[object] = None,
    ) -> None:
        """
        Initialize VehicleDetector with model weights and inference configuration.

        Args:
            model_path: Path to YOLO weights file (.pt).
            confidence_threshold: Minimum detection confidence threshold (0.0 to 1.0).
            device: Requested inference device ('0' for CUDA GPU, 'cpu' for CPU).
            classes: Subset of COCO class IDs to detect (defaults to all vehicles).
            imgsz: Input image inference dimension.
            model: Optional pre-loaded YOLO model instance for dependency injection.
        """
        self.model_path = str(model_path)
        self.confidence_threshold = float(confidence_threshold)
        self.device = resolve_device(device)
        self.imgsz = int(imgsz)

        if classes is not None:
            self.classes: Set[int] = set(classes)
        else:
            self.classes = set(COCO_VEHICLE_CLASSES.keys())

        self.class_names: Dict[int, str] = {
            cid: COCO_VEHICLE_CLASSES.get(cid, "Vehicle") for cid in self.classes
        }

        self._model = model

    @property
    def model(self) -> object:
        """Lazy loader for YOLO model instance."""
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
        return self._model

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Perform vehicle detection on an input BGR frame.

        Args:
            frame: 3-channel BGR numpy array representing a video frame.

        Returns:
            List of Detection objects matching target vehicle classes.
        """
        if frame is None or frame.size == 0:
            return []

        results = self.model(
            source=frame,
            conf=self.confidence_threshold,
            classes=list(self.classes),
            device=self.device,
            imgsz=self.imgsz,
            verbose=False,
        )[0]

        detections: List[Detection] = []
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy().astype(float)
            classes = results.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confs, classes):
                cid = int(cls_id)
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                cname = self.class_names.get(cid, "Vehicle")

                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=float(conf),
                        class_id=cid,
                        class_name=cname,
                    )
                )

        return detections
