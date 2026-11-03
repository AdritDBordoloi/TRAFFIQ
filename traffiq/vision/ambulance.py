"""
Shared emergency-vehicle (ambulance) detection helpers for TRAFFIQ.

Single source of truth for the Open Images V7 multi-label ambulance detector and
the box-matching rule that promotes a tracked vehicle to "Ambulance". Used by both
``traffiq.web.engine`` and the standalone ``live_unified.py`` pipeline.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch
from ultralytics.data.augment import LetterBox
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes

# Open Images V7 class id for "Ambulance".
OIV7_AMBULANCE_CLASS_ID = 6

# A tracked vehicle is promoted to ambulance when it overlaps a detection by at
# least one of these measures, or when its centroid lies inside the detection.
AMBULANCE_IOU_THRESHOLD = 0.20
AMBULANCE_IOMIN_THRESHOLD = 0.40

_letterbox_transformer = None


def get_letterbox_transformer() -> LetterBox:
    """Return a lazily created, shared 640px letterbox transformer."""
    global _letterbox_transformer
    if _letterbox_transformer is None:
        _letterbox_transformer = LetterBox(640)
    return _letterbox_transformer


def _intersection_area(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])
    return max(0, x_b - x_a) * max(0, y_b - y_a)


def _area(box: Sequence[float]) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])


def compute_box_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Intersection-over-Union between two ``(x1, y1, x2, y2)`` boxes."""
    inter = _intersection_area(box_a, box_b)
    if inter == 0:
        return 0.0
    union = float(_area(box_a) + _area(box_b) - inter)
    return inter / union if union > 0 else 0.0


def compute_box_iomin(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Intersection-over-Minimum-Area (robust to nested boxes of different scale)."""
    inter = _intersection_area(box_a, box_b)
    if inter == 0:
        return 0.0
    min_area = min(_area(box_a), _area(box_b))
    return inter / float(min_area) if min_area > 0 else 0.0


def point_in_box(cx: float, cy: float, box: Sequence[float]) -> bool:
    """True if point ``(cx, cy)`` lies inside the ``(x1, y1, x2, y2)`` box."""
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


def matches_ambulance_box(
    vehicle_box: Sequence[float], ambulance_box: Sequence[float], cx: float, cy: float
) -> bool:
    """Decide whether a tracked vehicle is the same object as an ambulance detection."""
    return (
        compute_box_iou(vehicle_box, ambulance_box) > AMBULANCE_IOU_THRESHOLD
        or compute_box_iomin(vehicle_box, ambulance_box) > AMBULANCE_IOMIN_THRESHOLD
        or point_in_box(cx, cy, ambulance_box)
    )


def detect_ambulances_oiv7(
    oiv7_model, frame: np.ndarray, device, conf_thres: float = 0.08, iou_thres: float = 0.45
) -> List[Tuple[np.ndarray, float]]:
    """
    Detect ambulances with YOLOv8s Open Images V7 using ``multi_label=True`` NMS.

    Standard single-label argmax over the 601 OIV7 classes lets 'Van' (564) or
    'Bus' (73) dominate and discard class 6; multi-label NMS keeps the ambulance.

    Returns a list of ``(xyxy_int_array, confidence)`` in original-frame coordinates.
    """
    h_orig, w_orig = frame.shape[:2]
    img_lb = get_letterbox_transformer()(image=frame)
    t_img = torch.from_numpy(img_lb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

    with torch.no_grad():
        preds = oiv7_model.model(t_img)

    nms_res = non_max_suppression(
        preds,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        classes=[OIV7_AMBULANCE_CLASS_ID],
        multi_label=True,
    )

    amb_boxes: List[Tuple[np.ndarray, float]] = []
    if len(nms_res) > 0 and len(nms_res[0]) > 0:
        scaled = scale_boxes(img_lb.shape[:2], nms_res[0][:, :4].clone(), (h_orig, w_orig))
        for b, conf in zip(scaled, nms_res[0][:, 4]):
            amb_boxes.append((b.cpu().numpy().astype(int), float(conf)))
    return amb_boxes
