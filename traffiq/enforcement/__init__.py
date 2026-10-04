"""
TRAFFIQ Enforcement Submodule:
Automated traffic rule enforcement components including signal state control,
red-light violation detection, license plate OCR, and e-challan generation.
"""

from traffiq.enforcement.challan import ChallanIssuer
from traffiq.enforcement.ocr import (
    INDIAN_RTO_REGEX,
    PlateExtractor,
    clamp_bbox,
    clean_plate_text,
    extract_plate_roi,
    is_valid_rto_plate,
    preprocess_plate,
    read_license_plate,
)
from traffiq.enforcement.signal import (
    SignalState,
    TrafficSignal,
    TrafficSignalController,
)
from traffiq.enforcement.violation import ViolationDetector, ViolationEvent

__all__ = [
    "TrafficSignal",
    "TrafficSignalController",
    "SignalState",
    "ViolationDetector",
    "ViolationEvent",
    "PlateExtractor",
    "ChallanIssuer",
    "clean_plate_text",
    "is_valid_rto_plate",
    "extract_plate_roi",
    "clamp_bbox",
    "preprocess_plate",
    "read_license_plate",
    "INDIAN_RTO_REGEX",
]
