"""
TRAFFIQ Vision Submodule:
Computer vision pipeline including camera streaming, vehicle detection,
ByteTrack trajectory tracking, virtual tripwires, and 4-way multi-approach processing.
"""

from traffiq.vision.camera import CameraStream
from traffiq.vision.detector import COCO_VEHICLE_CLASSES, Detection, VehicleDetector
from traffiq.vision.four_way import FourWayVideoProcessor
from traffiq.vision.tracker import (
    BoundedIDSet,
    TrackedVehicle,
    TrajectoryCache,
    VehicleTracker,
)
from traffiq.vision.tripwire import (
    Tripwire,
    TripwireCrossing,
    VirtualTripwire,
    calculate_centroid,
    check_line_crossing,
)

__all__ = [
    "CameraStream",
    "VehicleDetector",
    "Detection",
    "VehicleTracker",
    "TrackedVehicle",
    "VirtualTripwire",
    "Tripwire",
    "TripwireCrossing",
    "TrajectoryCache",
    "BoundedIDSet",
    "COCO_VEHICLE_CLASSES",
    "calculate_centroid",
    "check_line_crossing",
    "FourWayVideoProcessor",
]
