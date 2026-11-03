"""
TRAFFIQ Web Dashboard Streaming & Computer Vision Engine.
Wraps vehicle detection, tracking, stop-line violation enforcement, OCR,
and dual-model ambulance priority into a thread-safe streaming engine.
"""

from __future__ import annotations

import collections
import os
import queue
import sys
import threading
import time
from datetime import datetime
from dataclasses import dataclass, field, replace
from typing import Any, Collection, Deque, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from traffiq.database.connection import get_connection, register_sqlite_adapters
from traffiq.database.repository import DatabaseManager
from traffiq.enforcement.challan import ChallanIssuer
from traffiq.enforcement.ocr import PlateExtractor
from traffiq.enforcement.signal import TrafficSignal
from traffiq.enforcement.violation import ViolationDetector
from traffiq.vision.ambulance import detect_ambulances_oiv7, matches_ambulance_box
from traffiq.vision.tracker import BoundedIDSet, TrackedVehicle, VehicleTracker
from traffiq.vision.tripwire import VirtualTripwire

# Vehicle classes eligible for promotion to "Ambulance" (matches live_unified.py).
AMBULANCE_PROMOTABLE_CLASSES = frozenset({"Car", "Bus", "Truck"})
# Ambulance detector runs every N frames to save GPU time.
AMBULANCE_DETECT_CADENCE = 2
# Seconds an emergency green corridor stays active when started by a live detection.
EMERGENCY_HOLD_SECONDS = 20.0
# Fraction of frame height where an ambulance is considered to have cleared the junction.
EMERGENCY_EXIT_LINE_RATIO = 0.72
# Loop pacing for live cameras; video files are paced at their own FPS instead.
CAMERA_FRAME_INTERVAL = 0.033
MAX_PLAUSIBLE_VIDEO_FPS = 120.0
# Frames wider than this are scaled down before detection and streaming.
MAX_STREAM_WIDTH = 1280
JPEG_QUALITY = 70
# Video pacing: how much catching up to attempt before accepting a slower playback speed.
MAX_FRAMES_SKIPPED_PER_READ = 4
RESYNC_AFTER_SECONDS = 1.0
MAX_PACING_SLEEP_SECONDS = 0.05
# How often pipeline threads wake up to check for shutdown while no frame is arriving.
PIPELINE_WAIT_SECONDS = 0.1
AUTO_DEMO_IDLE_SECONDS = 60.0
# Detection needs fewer checks than video needs frames: a crossing is found by comparing a vehicle's
# position between checks, so none are missed. The cap leaves CPU time for smooth playback.
MAX_DETECTIONS_PER_SECOND = 30.0
# Each frame is held this long before display, so detection of that frame or a later one has finished
# by then. Boxes can be drawn where the vehicle is in that frame, rather than where it used to be.
DISPLAY_DELAY_SECONDS = 0.2
# Recent detection results kept for drawing boxes on the frames still waiting to be shown.
DETECTION_HISTORY_LENGTH = 16
# When detection falls behind the display delay, boxes keep moving at their last speed for at most this
# many detection intervals, so they neither freeze nor fly off.
MAX_BOX_EXTRAPOLATION = 2.0
# A reported rate drops to 0 when no frame has been shown for this long.
RATE_STALE_SECONDS = 1.0
# Ultralytics turns OpenCV threading off on import; scaling needs it back to keep up with 60 FPS.
OPENCV_THREADS = 4
# With inference on the GPU, PyTorch's default of one CPU thread per core starves the video threads.
# Measured on the 1080p60 test clip: 44 -> 60 FPS delivered, and detection 6/s -> 21/s.
TORCH_THREADS_WITH_GPU = 2
# Software decoding is faster than a GPU round trip for these simple codecs. Heavier ones, such as
# AV1 and VP9, need hardware decoding to play in real time.
SOFTWARE_DECODE_CODECS = frozenset({"h264", "avc1", "mp4v", "fmp4", "xvid", "mjpg"})
# Overlay colours (BGR), matching live_unified.py.
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_ORANGE = (255, 150, 0)


def describe_source(source: Union[int, str]) -> Dict[str, str]:
    """Human-readable description of a capture source, shown in the UI and on the HUD."""
    text = str(source).strip("'\"")
    if text.isdigit():
        return {"kind": "camera", "label": f"Webcam {text}"}
    return {"kind": "video", "label": os.path.basename(text)}


def plan_video_frame(now: float, start: float, next_index: int, interval: float) -> Tuple[float, int]:
    """Compare a video's timeline with the clock. Returns (seconds to wait, frames to skip); at most one is non-zero."""
    due = start + next_index * interval
    if now < due:
        return due - now, 0
    return 0.0, int((now - due) / interval)


def scale_to_stream_width(frame: np.ndarray) -> np.ndarray:
    """Scale a frame down to MAX_STREAM_WIDTH, keeping its proportions."""
    h, w = frame.shape[:2]
    if w <= MAX_STREAM_WIDTH:
        return frame
    return cv2.resize(frame, (MAX_STREAM_WIDTH, int(h * MAX_STREAM_WIDTH / w)), interpolation=cv2.INTER_LINEAR)


def fourcc_name(value: float) -> str:
    """Decode OpenCV's numeric CAP_PROP_FOURCC into a lower-case codec name such as 'h264' or 'av01'."""
    code = int(value)
    return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ").lower()


class RateMeter:
    """Events per second over a sliding window; steadier than timing a single frame."""

    def __init__(self, window_seconds: float = 1.0) -> None:
        self._window = window_seconds
        self._ticks: Deque[float] = collections.deque()

    def tick(self, now: float) -> float:
        self._ticks.append(now)
        while now - self._ticks[0] > self._window:
            self._ticks.popleft()
        elapsed = now - self._ticks[0]
        return (len(self._ticks) - 1) / elapsed if elapsed > 0 else 0.0


@dataclass(frozen=True)
class DetectionOverlay:
    """The newest detection results. Replaced whole, so the render thread never sees a half-updated set."""

    generation: int = 0
    frame_number: int = 0
    tracked: List[TrackedVehicle] = field(default_factory=list)
    counted_ids: Collection[int] = ()
    amb_boxes: List[Tuple[np.ndarray, float]] = field(default_factory=list)


@dataclass(frozen=True)
class QueuedFrame:
    """A frame waiting for its display time."""

    frame: np.ndarray
    frame_number: int
    generation: int
    display_at: float


def _blend_vehicle(before: TrackedVehicle, after: TrackedVehicle, t: float) -> TrackedVehicle:
    def blend(a: Tuple[int, ...], b: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))

    return replace(after, bbox=blend(before.bbox, after.bbox), centroid=blend(before.centroid, after.centroid))


def overlay_for_frame(history: Sequence[DetectionOverlay], frame_number: int) -> DetectionOverlay:
    """
    Work out where the boxes belong on one frame from the detections around it (oldest first).

    A frame that was analysed gets its exact boxes. A frame between two analysed frames gets each box
    blended between them. A newer frame continues each box at its last speed, up to a limit.
    """
    if not history:
        return DetectionOverlay()

    later_index = next((i for i, d in enumerate(history) if d.frame_number >= frame_number), None)
    if later_index == 0 or len(history) == 1:
        return history[0]
    after_index = len(history) - 1 if later_index is None else later_index
    before, after = history[after_index - 1], history[after_index]

    t = (frame_number - before.frame_number) / (after.frame_number - before.frame_number)
    t = min(t, MAX_BOX_EXTRAPOLATION)
    before_by_id = {v.track_id: v for v in before.tracked}
    after_ids = {v.track_id for v in after.tracked}

    vehicles: List[TrackedVehicle] = []
    for vehicle in after.tracked:
        previous = before_by_id.get(vehicle.track_id)
        if previous is not None:
            vehicles.append(_blend_vehicle(previous, vehicle, t))
        elif t >= 0.5:  # newly detected: appears at the frame nearer its first detection
            vehicles.append(vehicle)
    if t < 0.5:  # lost by the later detection: stays until the frame nearer that detection
        vehicles.extend(v for v in before.tracked if v.track_id not in after_ids)

    return replace(after, frame_number=frame_number, tracked=vehicles)


def is_enforceable_crossing(crossed: bool, signal_state: str, is_emergency: bool) -> bool:
    """A stop-line crossing is a red-light violation unless the vehicle is an identified emergency vehicle."""
    return bool(crossed) and signal_state == "RED" and not is_emergency


def evidence_url(path: str, root: str) -> str:
    """
    Map an evidence file to its ``/evidence/<path relative to root>`` URL.

    Both sides are resolved with ``realpath`` first (``ChallanIssuer`` stores resolved paths, so
    symlinks / Windows 8.3 short names would otherwise yield ``../..`` garbage). Returns "" for
    empty paths and for files outside ``root`` (they could not be served anyway).
    """
    if not path:
        return ""
    try:
        rel = os.path.relpath(os.path.realpath(path), os.path.realpath(root))
    except ValueError:  # different drive on Windows
        return ""
    if rel == os.pardir or rel.startswith(os.pardir + os.sep):
        return ""
    return "/evidence/" + rel.replace("\\", "/")


class DashboardEngine:
    """
    Background vision engine orchestrating live video frame capture,
    AI inference, HUD overlay generation, and event dispatch for the web dashboard.
    """

    def __init__(
        self,
        source: Union[int, str] = 0,
        model_path: str = "yolov8s.pt",
        oiv7_path: str = "yolov8s-oiv7.pt",
        db_path: str = "traffiq_enforcement.db",
        violations_dir: str = "violations",
        confidence_threshold: float = 0.25,
        tripwire_ratio: float = 0.55,
        fine_amount: int = 1000,
        signal_duration: float = 8.0,
        device: str = "0",
        auto_demo: bool = False,
        start_thread: bool = True,
        ambulance_conf: float = 0.08,
    ) -> None:
        register_sqlite_adapters()
        self.source = source
        self.model_path = model_path
        self.oiv7_path = oiv7_path
        self.db_path = db_path
        self.violations_dir = violations_dir
        self.confidence_threshold = float(confidence_threshold)
        self.tripwire_ratio = float(tripwire_ratio)
        self.fine_amount = int(fine_amount)
        self.signal_duration = float(signal_duration)
        self.auto_demo = bool(auto_demo)
        self.ambulance_conf = float(ambulance_conf)

        os.makedirs(self.violations_dir, exist_ok=True)

        # Device determination
        # CLI-style device ("0" / "cpu") -> ultralytics string + a valid torch.device ("cuda:0")
        requested = str(device).strip()
        use_cuda = requested.isdigit() and torch.cuda.is_available()
        self.device_str = requested if use_cuda else "cpu"
        self.device = torch.device(f"cuda:{requested}" if use_cuda else "cpu")

        # Vision and enforcement components
        self.db_manager = DatabaseManager(db_path=self.db_path)
        self.signal = TrafficSignal(cycle_duration=self.signal_duration)
        self.violation_detector = ViolationDetector(fine_amount=self.fine_amount)
        self.plate_extractor = PlateExtractor(gpu=(self.device_str != "cpu"))
        self.challan_issuer = ChallanIssuer(
            db_manager=self.db_manager,
            output_dir=self.violations_dir,
            fine_amount=self.fine_amount,
        )

        # Primary tracker
        self.tracker: Optional[VehicleTracker] = None
        self.oiv7_model: Optional[YOLO] = None

        # Threading state
        self._lock = threading.Lock()
        self._running = False
        self._threads: List[threading.Thread] = []
        self._latest_jpeg: Optional[bytes] = None
        self._jpeg_seq = 0
        self._jpeg_ready = threading.Condition(self._lock)
        self._last_render_time = 0.0
        self._pending_source: Optional[Union[int, str]] = None

        # Pipeline: capture -> prepare -> (detect, display queue -> render). Each stage takes the newest
        # frame from a one-frame slot, so a slow stage drops frames instead of delaying the ones before it.
        self._frames = threading.Condition()
        self._generation = 0  # increases on every source switch
        self._raw_frame: Optional[np.ndarray] = None
        self._raw_seq = 0
        self._raw_generation = 0
        self._raw_display_at = 0.0
        self._display_queue: Deque[QueuedFrame] = collections.deque()
        self._detect_frame: Optional[np.ndarray] = None
        self._detect_seq = 0
        self._detect_generation = 0
        # Replaced whole on every update, so the render thread can read it without a lock
        self._detection_history: Tuple[DetectionOverlay, ...] = ()
        self._violation_queue: "queue.Queue[Tuple[int, np.ndarray]]" = queue.Queue()

        # Statistics & Events (Bounded)
        self.stats: Dict[str, Any] = {
            "vehicles": 0,
            "violations": 0,
            "emergencies": 0,
            "total_fines": 0,
            "fps": 0.0,
            "detect_fps": 0.0,
        }
        self.events: Deque[Dict[str, Any]] = collections.deque(maxlen=100)
        self._event_counter = 0
        self._last_event_time = time.time()

        # Emergency Priority State
        self.is_emergency_locked: bool = False
        self.emergency_end_time: float = 0.0
        self.emergency_vehicle_track_id: Optional[int] = None
        # Tracks already promoted to ambulance; bounded so memory cannot grow without limit,
        # and never pruned per-frame so a brief detection dropout cannot re-trigger the corridor.
        self.active_emergency_tracks = BoundedIDSet(max_size=1000)
        # Unique vehicle track ids seen since the last stats reset (feeds the Vehicles KPI)
        self._seen_vehicle_ids = BoundedIDSet(max_size=10000)

        # Generate default placeholder frame
        self._create_placeholder_frame("TRAFFIQ LIVE FEED - Initializing...")

        if start_thread:
            self.start()

    def _create_placeholder_frame(self, message: str) -> None:
        """Create a synthetic 1280x720 standby/demo frame."""
        img = np.full((720, 1280, 3), 25, dtype=np.uint8)
        # Grid lines
        for y in range(0, 720, 40):
            cv2.line(img, (0, y), (1280, y), (35, 35, 35), 1)
        for x in range(0, 1280, 40):
            cv2.line(img, (x, 0), (x, 720), (35, 35, 35), 1)

        # Stop-line indication
        stop_y = int(720 * self.tripwire_ratio)
        cv2.line(img, (0, stop_y), (1280, stop_y), (60, 60, 60), 2)
        cv2.putText(img, "STOP LINE [y=55%]", (20, stop_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)

        # Center banner
        cv2.rectangle(img, (340, 310), (940, 410), (15, 23, 42), -1)
        cv2.rectangle(img, (340, 310), (940, 410), (56, 189, 248), 2)
        cv2.putText(img, "TRAFFIQ VISION ENGINE", (380, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, message, (360, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (56, 189, 248), 1)

        ret, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ret:
            self._publish_jpeg(buf.tobytes())

    def _publish_jpeg(self, jpeg: bytes) -> None:
        with self._jpeg_ready:
            self._latest_jpeg = jpeg
            self._jpeg_seq += 1
            self._jpeg_ready.notify_all()

    def start(self) -> None:
        """Start the capture, render, detection and violation threads."""
        if not self._running:
            self._running = True
            cv2.setNumThreads(OPENCV_THREADS)
            if self.device.type == "cuda":
                torch.set_num_threads(TORCH_THREADS_WITH_GPU)
            stages = {
                "Capture": self._capture_loop,
                "Prepare": self._prepare_loop,
                "Render": self._render_loop,
                "Detect": self._detect_loop,
                "Violations": self._violation_loop,
            }
            self._threads = [
                threading.Thread(target=target, daemon=True, name=f"Dashboard{name}Thread")
                for name, target in stages.items()
            ]
            for thread in self._threads:
                thread.start()

    def stop(self) -> None:
        """Stop background execution."""
        self._running = False
        with self._frames:
            self._frames.notify_all()
        for thread in self._threads:
            if thread.is_alive():
                thread.join(timeout=2.0)

    def wait_for_frame(self, last_seq: int, timeout: float = 1.0) -> Tuple[int, bytes]:
        """Block until a JPEG newer than last_seq exists, or timeout. Returns (sequence number, JPEG bytes)."""
        with self._jpeg_ready:
            self._jpeg_ready.wait_for(lambda: self._jpeg_seq != last_seq, timeout=timeout)
            return self._jpeg_seq, self._latest_jpeg or b""

    def _signal_state(self) -> str:
        return "GREEN" if self.is_emergency_locked else self.signal.state

    def _update_signal(self, now: float) -> str:
        """Advance the signal timer, end an expired corridor, and return the current signal state."""
        with self._lock:
            if self.is_emergency_locked:
                if now >= self.emergency_end_time:
                    self.is_emergency_locked = False
                    self._push_event("emergency", {
                        "vehicle_type": "Ambulance",
                        "status": "cleared",
                    })
            else:
                self.signal.update(current_time=now)
            return self._signal_state()

    def get_source(self) -> Dict[str, str]:
        """Describe the capture source currently selected."""
        with self._lock:
            return describe_source(self.source)

    def set_source(self, source: Union[int, str]) -> Dict[str, str]:
        """Switch to another camera or video file; the capture thread applies it on its next frame."""
        with self._lock:
            self.source = source
            self._pending_source = source
            return describe_source(source)

    def get_latest_frame(self) -> bytes:
        """Retrieve the most recent JPEG bytes."""
        with self._lock:
            if self._latest_jpeg is not None:
                return self._latest_jpeg
        # Fallback standby frame
        self._create_placeholder_frame("TRAFFIQ STANDBY")
        with self._lock:
            return self._latest_jpeg or b""

    def get_state(self) -> Dict[str, Any]:
        """Return instantaneous state dictionary for frontend polling."""
        with self._lock:
            now = time.time()
            rem = self.signal.time_remaining(current_time=now) if not self.is_emergency_locked else 0.0
            is_streaming = now - self._last_render_time < RATE_STALE_SECONDS
            return {
                "stats": {
                    "vehicles": int(self.stats["vehicles"]),
                    "violations": int(self.stats["violations"]),
                    "emergencies": int(self.stats["emergencies"]),
                    "total_fines": int(self.stats["total_fines"]),
                    "fps": round(float(self.stats["fps"]), 1) if is_streaming else 0.0,
                    "detect_fps": round(float(self.stats["detect_fps"]), 1) if is_streaming else 0.0,
                },
                "signal": {
                    "state": "GREEN" if self.is_emergency_locked else self.signal.state,
                    "time_remaining": round(float(rem), 1),
                    "is_emergency_locked": bool(self.is_emergency_locked),
                    "emergency_track_id": self.emergency_vehicle_track_id if self.is_emergency_locked else None,
                },
                "emergency_active": bool(self.is_emergency_locked),
                "source": describe_source(self.source),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

    def get_events(self, since_id: int = 0) -> List[Dict[str, Any]]:
        """Return new events occurred after since_id."""
        with self._lock:
            return [ev for ev in self.events if ev["id"] > since_id]

    def toggle_signal(self) -> Dict[str, Any]:
        """Manually toggle traffic signal if not locked by emergency."""
        with self._lock:
            if self.is_emergency_locked:
                return {"success": False, "message": "Signal locked during emergency corridor."}
            new_state = self.signal.toggle()
            return {"success": True, "state": new_state}

    def reset_stats(self) -> Dict[str, Any]:
        """Reset live dashboard counters."""
        with self._lock:
            self.stats["vehicles"] = 0
            self.stats["violations"] = 0
            self.stats["emergencies"] = 0
            self.stats["total_fines"] = 0
            self._seen_vehicle_ids.clear()
            return {"success": True}

    def _push_event(self, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Store new event in bounded deque."""
        self._event_counter += 1
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ev = {
            "id": int(self._event_counter),
            "type": str(event_type),
            "timestamp": now_str,
            "payload": payload,
        }
        self.events.append(ev)
        self._last_event_time = time.time()
        return ev

    def _publish_violation(
        self,
        challan_id: int,
        track_id: int,
        plate: str,
        owner: str,
        contact: str,
        model: str,
        image_path: str,
        plate_image_path: str = "",
    ) -> Dict[str, Any]:
        """Update counters and push the ``violation`` event consumed by the dashboard."""
        payload = {
            "challan_id": int(challan_id),
            "track_id": int(track_id),
            "plate": plate,
            "owner": owner,
            "contact": contact,
            "model": model,
            "fine": int(self.fine_amount),
            "image_url": evidence_url(image_path, self.violations_dir),
            "plate_url": evidence_url(plate_image_path, self.violations_dir),
        }
        with self._lock:
            self.stats["violations"] += 1
            self.stats["total_fines"] += int(self.fine_amount)
            return self._push_event("violation", payload)

    def record_violation(
        self, track_id: int, plate_text: str, vehicle_crop: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """
        Issue a real e-challan for a live red-light violation using the actual
        tracked vehicle id, the real evidence crop and the OCR-read plate. Unreadable
        or unregistered plates are recorded as such, never substituted with another vehicle.
        """
        challan = self.challan_issuer.issue_challan(
            track_id=int(track_id),
            vehicle_crop=vehicle_crop,
            detected_plate=plate_text or "UNKNOWN",
            fine_amount=self.fine_amount,
        )
        return self._publish_violation(
            challan_id=challan["challan_id"],
            track_id=challan["track_id"],
            plate=challan["plate_number"],
            owner=challan["owner_name"],
            contact=challan["contact_number"],
            model=challan["vehicle_model"],
            image_path=challan["evidence_path"],
        )

    def trigger_demo_violation(self, plate: Optional[str] = None) -> Dict[str, Any]:
        """
        Inject a clearly labelled *demo* violation (synthetic evidence, sample owner)
        through the real database path so the dashboard can be shown without a camera.
        """
        sample_plates = ["AS01AB1234", "DL04CA9999", "MH02DZ4567", "KA05MJ8821"]
        chosen_plate = plate or sample_plates[self._event_counter % len(sample_plates)]

        owner_info = self.db_manager.lookup_vehicle(chosen_plate) or {
            "owner": "Rajesh Sharma",
            "phone": "+91 98765 43210",
            "model": "Hyundai Creta",
            "city": "Guwahati",
        }

        track_id = int(800 + (self._event_counter % 100))
        ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ts_slug = datetime.now().strftime("%Y%m%d_%H%M%S")

        v_dir = os.path.join(self.violations_dir, f"vehicle_ID{track_id}_{ts_slug}")
        os.makedirs(v_dir, exist_ok=True)
        full_img_path = os.path.join(v_dir, "vehicle_image.jpg")
        plate_img_path = os.path.join(v_dir, "plate_crop.jpg")
        receipt_path = os.path.join(v_dir, "challan.txt")

        # Synthetic evidence: stylised car + plate crop
        sim_frame = np.full((360, 480, 3), 45, dtype=np.uint8)
        cv2.rectangle(sim_frame, (80, 100), (400, 280), (30, 40, 180), -1)
        cv2.rectangle(sim_frame, (120, 60), (360, 160), (20, 30, 120), -1)
        cv2.putText(sim_frame, f"ID #{track_id} - DEMO RED LIGHT RUNNER", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        cv2.imwrite(full_img_path, sim_frame)

        plate_crop = np.full((80, 240, 3), 245, dtype=np.uint8)
        cv2.rectangle(plate_crop, (2, 2), (238, 78), (0, 0, 0), 2)
        cv2.putText(plate_crop, chosen_plate, (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
        cv2.imwrite(plate_img_path, plate_crop)

        owner = owner_info.get("owner", "Demo Driver")
        phone = owner_info.get("phone", "+91 90000 00000")
        model = owner_info.get("model", "Test Vehicle")
        challan_id = self.db_manager.record_challan(
            track_id=track_id,
            plate_number=chosen_plate,
            owner_name=owner,
            violation_type="Red Light Signal Violation (Demo)",
            fine_amount=int(self.fine_amount),
            evidence_path=full_img_path,
            timestamp=ts_str,
            contact_number=phone,
            vehicle_model=model,
            full_evidence_path=full_img_path,
            crop_evidence_path=plate_img_path,
        )

        with open(receipt_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("       TRAFFIQ AUTOMATED TRAFFIC ENFORCEMENT NOTICE (DEMO)\n")
            f.write("=" * 60 + "\n")
            f.write(f"Challan Number     : TRFQ-{challan_id:06d}\n")
            f.write(f"Violation DateTime : {ts_str}\n")
            f.write(f"Vehicle Track ID   : #{track_id}\n")
            f.write(f"Detected Plate     : {chosen_plate}\n")
            f.write(f"Registered Owner   : {owner}\n")
            f.write(f"Contact Number     : {phone}\n")
            f.write(f"Fine Amount Due    : INR {self.fine_amount}\n")
            f.write("=" * 60 + "\n")

        return self._publish_violation(
            challan_id, track_id, chosen_plate, owner, phone, model, full_img_path, plate_img_path
        )

    def _start_emergency(self, duration: float, track_id: int, vehicle_type: str) -> Dict[str, Any]:
        """Lock the signal GREEN (emergency corridor), log it to SQLite and push an event."""
        with self._lock:
            self.is_emergency_locked = True
            self.emergency_end_time = time.time() + float(duration)
            self.emergency_vehicle_track_id = int(track_id)
            self.stats["emergencies"] += 1

        conn = None
        try:
            conn = get_connection(self.db_path)
            conn.execute(
                """
                INSERT INTO emergency_events (timestamp, track_id, vehicle_type, override_duration_sec, status)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    int(track_id),
                    vehicle_type,
                    float(duration),
                    "active",
                ),
            )
            conn.commit()
        except Exception as exc:
            print(f"[WARN] Could not log emergency event: {exc}")
        finally:
            if conn is not None:
                conn.close()

        with self._lock:
            return self._push_event(
                "emergency",
                {
                    "track_id": int(track_id),
                    "vehicle_type": "Ambulance",
                    "status": "active",
                    "duration": float(duration),
                },
            )

    def trigger_demo_ambulance(self, duration: float = 15.0) -> Dict[str, Any]:
        """Trigger a simulated emergency green corridor (demo button / API)."""
        track_id = 990 + (self._event_counter % 10)
        return self._start_emergency(duration, track_id, "Ambulance (Demo)")

    def _new_tracker(self) -> VehicleTracker:
        return VehicleTracker(
            model_path=self.model_path,
            confidence_threshold=self.confidence_threshold,
            device=self.device_str,
        )

    def _open_capture(self, source: Union[int, str]) -> Tuple[Optional[cv2.VideoCapture], bool]:
        """Open a camera index or video file. Returns (capture, is_video_file)."""
        src_str = str(source).strip("'\"")
        cap: Optional[cv2.VideoCapture] = None
        is_video_file = False
        try:
            if src_str.isdigit():
                cap = cv2.VideoCapture(int(src_str), cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                if sys.platform.startswith("win"):
                    try:
                        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                    except Exception:
                        pass
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            else:
                is_video_file = os.path.isfile(src_str)
                cap = self._open_video_file(src_str)
        except Exception as e:
            print(f"[WARN] Failed opening video source: {e}")
        if cap is None or not cap.isOpened():
            print(f"[WARN] Video source unavailable: {src_str}")
        return cap, is_video_file

    @staticmethod
    def _open_video_file(path: str) -> cv2.VideoCapture:
        """Open a video, using hardware decoding for codecs too heavy to decode in software in real time."""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened() or fourcc_name(cap.get(cv2.CAP_PROP_FOURCC)) in SOFTWARE_DECODE_CODECS:
            return cap
        accelerated = cv2.VideoCapture(
            path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY]
        )
        if accelerated.isOpened():
            cap.release()
            return accelerated
        accelerated.release()
        return cap

    @staticmethod
    def _frame_interval(cap: Optional[cv2.VideoCapture], is_video_file: bool) -> float:
        """Seconds per frame: a video's own FPS so it plays at natural speed, else the camera pacing."""
        if cap is None or not is_video_file:
            return CAMERA_FRAME_INTERVAL
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or fps > MAX_PLAUSIBLE_VIDEO_FPS:
            return CAMERA_FRAME_INTERVAL
        return 1.0 / fps

    def _reset_for_new_source(self) -> None:
        """
        Forget everything tied to the previous source. Track ids restart in a new video, so old
        ids would otherwise suppress its violations and undercount its vehicles. Challans already
        saved to the database are kept.
        """
        if self.tracker is not None:
            self.tracker = self._new_tracker()
        self.violation_detector = ViolationDetector(fine_amount=self.fine_amount)
        self.active_emergency_tracks.clear()
        self.reset_stats()
        with self._lock:
            if self.is_emergency_locked:
                self.is_emergency_locked = False
                self._push_event("emergency", {"vehicle_type": "Ambulance", "status": "cleared"})

    def _draw_overlays(
        self,
        frame: np.ndarray,
        stop_y: int,
        exit_y: int,
        sig_state: str,
        tracked_vehicles: List[TrackedVehicle],
        counted_ids: Collection[int],
        amb_boxes: List[Tuple[np.ndarray, float]],
    ) -> None:
        """
        Draw both lines and the state-coloured boxes from live_unified.py. Signal, counters and the
        emergency alert are shown by the web page, whose translucent header sits over the video.
        """
        w = frame.shape[1]
        line_col = COLOR_RED if sig_state == "RED" else COLOR_GREEN

        cv2.line(frame, (0, stop_y), (w, stop_y), line_col, 3)
        cv2.putText(frame, f"STOP LINE [{sig_state}]", (10, stop_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, line_col, 2)
        cv2.line(frame, (0, exit_y), (w, exit_y), COLOR_YELLOW, 2)
        cv2.putText(frame, "CLEARANCE EXIT LINE", (10, exit_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_YELLOW, 1)

        for tv in tracked_vehicles:
            tid = int(tv.track_id)
            bx1, by1, bx2, by2 = tv.bbox
            pct = int(tv.confidence * 100)
            if tid in self.active_emergency_tracks:
                b_col, lbl, thickness = COLOR_RED, f"EMERGENCY: Ambulance #{tid}", 3
            elif self.violation_detector.is_violation(tid):
                b_col, lbl, thickness = COLOR_RED, f"VIOLATION #{tid}", 2
            elif tid in counted_ids:
                b_col, lbl, thickness = COLOR_GREEN, f"ID:{tid} {tv.class_name} {pct}%", 2
            else:
                b_col, lbl, thickness = COLOR_ORANGE, f"ID:{tid} {tv.class_name} {pct}%", 2
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), b_col, thickness)
            cv2.circle(frame, (int(tv.centroid[0]), int(tv.centroid[1])), 4, COLOR_RED, -1)
            cv2.putText(frame, lbl, (bx1, max(20, by1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, b_col, 2)

        # Ambulance detections with no tracked vehicle underneath: shown, but they do not hold the signal.
        for abox, amb_conf in amb_boxes:
            if any(
                matches_ambulance_box(tv.bbox, abox, int(tv.centroid[0]), int(tv.centroid[1]))
                for tv in tracked_vehicles
                if tv.class_name in AMBULANCE_PROMOTABLE_CLASSES
            ):
                continue
            ax1, ay1, ax2, ay2 = (int(v) for v in abox[:4])
            cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), COLOR_RED, 1)
            cv2.putText(frame, f"Ambulance? {int(amb_conf * 100)}%", (ax1, max(20, ay1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_RED, 1)

    def _load_models(self) -> None:
        try:
            if os.path.exists(self.model_path):
                self.tracker = self.tracker or self._new_tracker()
            if os.path.exists(self.oiv7_path):
                self.oiv7_model = YOLO(self.oiv7_path)
                self.oiv7_model.to(self.device)
        except Exception as e:
            print(f"[WARN] Error loading models in DashboardEngine: {e}")

    def _maybe_trigger_auto_demo(self) -> None:
        if self.auto_demo and (time.time() - self._last_event_time > AUTO_DEMO_IDLE_SECONDS):
            self.trigger_demo_violation()

    def _capture_loop(self) -> None:
        """Pipeline stage 1: decode frames at the source's own pace, dropping frames if decoding falls behind."""
        with self._lock:
            self._pending_source = None
            initial_source = self.source
        cap, is_video_file = self._open_capture(initial_source)
        frame_interval = self._frame_interval(cap, is_video_file)
        playback_start = time.perf_counter()
        next_index = 0

        while self._running:
            # Apply a source switch requested from the dashboard
            with self._lock:
                pending_source, self._pending_source = self._pending_source, None
            if pending_source is not None:
                if cap is not None:
                    cap.release()
                with self._frames:
                    self._generation += 1
                    self._raw_frame = None
                    self._display_queue.clear()
                    self._frames.notify_all()
                self._create_placeholder_frame(f"Loading {describe_source(pending_source)['label'][:40]}...")
                cap, is_video_file = self._open_capture(pending_source)
                frame_interval = self._frame_interval(cap, is_video_file)
                playback_start = time.perf_counter()
                next_index = 0
                continue

            if cap is None or not cap.isOpened():
                self._create_placeholder_frame("Video source unavailable - choose another source")
                time.sleep(0.05)
                self._maybe_trigger_auto_demo()
                continue

            if is_video_file:
                now = time.perf_counter()
                wait, skip = plan_video_frame(now, playback_start, next_index, frame_interval)
                if wait > 0:
                    # Short sleeps so a source switch is still picked up quickly
                    time.sleep(min(wait, MAX_PACING_SLEEP_SECONDS))
                    continue
                if skip * frame_interval > RESYNC_AFTER_SECONDS:
                    # Too far behind to catch up: continue from here rather than stalling on skips
                    playback_start = now - next_index * frame_interval
                    skip = 0
                # grab() skips the costly colour conversion and GPU-to-CPU copy of a full read()
                for _ in range(min(skip, MAX_FRAMES_SKIPPED_PER_READ)):
                    if not cap.grab():
                        break
                    next_index += 1

            ret, frame = cap.read()
            if not ret or frame is None:
                if is_video_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    playback_start = time.perf_counter()
                    next_index = 0
                time.sleep(frame_interval)
                continue
            # A video's timeline sets its display time, which also evens out uneven decode times
            source_time = playback_start + next_index * frame_interval if is_video_file else time.perf_counter()
            next_index += 1

            with self._frames:
                self._raw_frame = frame
                self._raw_seq += 1
                self._raw_generation = self._generation
                self._raw_display_at = source_time + DISPLAY_DELAY_SECONDS
                self._frames.notify_all()
            self._maybe_trigger_auto_demo()

    def _prepare_loop(self) -> None:
        """Pipeline stage 2: scale each frame, then hand it to detection and to the display queue."""
        seen_seq = 0
        while self._running:
            with self._frames:
                self._frames.wait_for(lambda: self._raw_seq != seen_seq or not self._running, timeout=PIPELINE_WAIT_SECONDS)
                if self._raw_seq == seen_seq or self._raw_frame is None:
                    continue
                raw, seen_seq, generation, display_at = (
                    self._raw_frame, self._raw_seq, self._raw_generation, self._raw_display_at,
                )

            frame = scale_to_stream_width(raw)
            with self._frames:
                if generation != self._generation:
                    continue
                self._detect_frame = frame
                self._detect_seq += 1
                self._detect_generation = generation
                self._display_queue.append(QueuedFrame(frame, self._detect_seq, generation, display_at))
                self._frames.notify_all()

    def _next_frame_to_show(self) -> Optional[QueuedFrame]:
        """Wait for a queued frame's display time. If rendering fell behind, jump to the newest frame due."""
        with self._frames:
            self._frames.wait_for(lambda: self._display_queue or not self._running, timeout=PIPELINE_WAIT_SECONDS)
            if not self._display_queue:
                return None
            wait = self._display_queue[0].display_at - time.perf_counter()
            if wait <= 0:
                now = time.perf_counter()
                while len(self._display_queue) > 1 and self._display_queue[1].display_at <= now:
                    self._display_queue.popleft()
                return self._display_queue.popleft()
        # Short sleeps so a source switch still takes effect quickly
        time.sleep(min(wait, MAX_PACING_SLEEP_SECONDS))
        return None

    def _render_loop(self) -> None:
        """Pipeline stage 3: at each frame's display time, draw its boxes and JPEG-encode it for the browser."""
        meter = RateMeter()
        while self._running:
            queued = self._next_frame_to_show()
            if queued is None:
                continue

            now = time.time()
            sig_state = self._update_signal(now)
            h = queued.frame.shape[0]
            history = [d for d in self._detection_history if d.generation == queued.generation]
            overlay = overlay_for_frame(history, queued.frame_number)
            canvas = queued.frame.copy()  # the detection stage may still be reading the clean frame
            self._draw_overlays(
                canvas,
                int(h * self.tripwire_ratio),
                int(h * EMERGENCY_EXIT_LINE_RATIO),
                sig_state,
                overlay.tracked,
                overlay.counted_ids,
                overlay.amb_boxes,
            )
            ret, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ret:
                self._publish_jpeg(buf.tobytes())
            rate = meter.tick(now)
            with self._lock:
                self.stats["fps"] = rate
                self._last_render_time = now

    def _detect_loop(self) -> None:
        """Pipeline stage 4: analyse only the newest frame, so slow inference skips frames instead of delaying video."""
        self._load_models()

        with self._frames:
            generation = self._generation
        seen_seq = 0
        tripwire: Optional[VirtualTripwire] = None
        frame_idx = 0
        cached_amb_boxes: List[Tuple[np.ndarray, float]] = []
        meter = RateMeter()

        while self._running:
            with self._frames:
                self._frames.wait_for(
                    lambda: self._detect_seq != seen_seq or self._generation != generation or not self._running,
                    timeout=PIPELINE_WAIT_SECONDS,
                )
                latest_generation = self._generation
                frame, frame_seq, frame_generation = self._detect_frame, self._detect_seq, self._detect_generation

            if latest_generation != generation:
                generation = latest_generation
                tripwire = None
                frame_idx = 0
                cached_amb_boxes = []
                self._detection_history = ()
                self._reset_for_new_source()
            if frame_seq == seen_seq or frame is None or frame_generation != generation:
                continue
            seen_seq = frame_seq
            detection_start = time.perf_counter()

            h = frame.shape[0]
            if tripwire is None:
                tripwire = VirtualTripwire(y_position=int(h * self.tripwire_ratio))
            frame_idx += 1

            tracked_vehicles: List[TrackedVehicle] = []
            if self.tracker is not None:
                try:
                    tracked_vehicles = self.tracker.track(frame, frame_idx=frame_idx)
                except Exception:
                    tracked_vehicles = []

            if self.oiv7_model is not None and (frame_idx % AMBULANCE_DETECT_CADENCE == 0):
                try:
                    cached_amb_boxes = detect_ambulances_oiv7(
                        self.oiv7_model,
                        frame,
                        device=self.device,
                        conf_thres=self.ambulance_conf,
                    )
                except Exception:
                    cached_amb_boxes = []

            try:
                self._evaluate_vehicles(frame, tracked_vehicles, tripwire, cached_amb_boxes)
            except Exception as exc:  # one bad frame must not stop detection for the rest of the session
                print(f"[WARN] Could not evaluate frame {frame_idx}: {exc}")
            self._detection_history = (
                *self._detection_history[-(DETECTION_HISTORY_LENGTH - 1):],
                DetectionOverlay(generation, frame_seq, tracked_vehicles, tripwire.counted_ids, cached_amb_boxes),
            )

            rate = meter.tick(time.time())
            with self._lock:
                self.stats["detect_fps"] = rate
            spare_time = 1.0 / MAX_DETECTIONS_PER_SECOND - (time.perf_counter() - detection_start)
            if spare_time > 0:
                time.sleep(spare_time)

    def _evaluate_vehicles(
        self,
        frame: np.ndarray,
        tracked_vehicles: List[TrackedVehicle],
        tripwire: VirtualTripwire,
        amb_boxes: List[Tuple[np.ndarray, float]],
    ) -> None:
        """Count vehicles, promote ambulances, clear corridors and queue red-light violations."""
        exit_y = int(frame.shape[0] * EMERGENCY_EXIT_LINE_RATIO)
        sig_state = self._signal_state()

        for tv in tracked_vehicles:
            if int(tv.track_id) not in self._seen_vehicle_ids:
                self._seen_vehicle_ids.add(int(tv.track_id))
                with self._lock:
                    self.stats["vehicles"] += 1

        for tv in tracked_vehicles:
            tid = int(tv.track_id)
            cx, cy = int(tv.centroid[0]), int(tv.centroid[1])
            bbox = (int(tv.bbox[0]), int(tv.bbox[1]), int(tv.bbox[2]), int(tv.bbox[3]))

            # 1. Ambulance check (promote a track once, using the shared matching rule)
            is_amb = tid in self.active_emergency_tracks
            if not is_amb and tv.class_name in AMBULANCE_PROMOTABLE_CLASSES:
                for abox, _amb_conf in amb_boxes:
                    if matches_ambulance_box(bbox, abox, cx, cy):
                        is_amb = True
                        self.active_emergency_tracks.add(tid)
                        if not self.is_emergency_locked:
                            self._start_emergency(EMERGENCY_HOLD_SECONDS, tid, "Ambulance")
                        break

            # Emergency exit clearance line
            if is_amb and cy >= exit_y:
                with self._lock:
                    if self.is_emergency_locked and self.emergency_vehicle_track_id in (tid, None):
                        self.is_emergency_locked = False
                        self._push_event("emergency", {
                            "track_id": tid,
                            "vehicle_type": "Ambulance",
                            "status": "cleared",
                        })

            # 2. Stop-Line Crossing Evaluation
            crossed = tripwire.update_track(tid, (cx, cy))
            if is_enforceable_crossing(crossed, sig_state, is_amb):
                viol = self.violation_detector.evaluate_crossing(
                    track_id=tid,
                    crossed=True,
                    signal_state="RED",
                    bbox=bbox,
                    frame=frame,
                )
                if viol:
                    self._violation_queue.put((tid, viol.vehicle_crop))

    def _violation_loop(self) -> None:
        """Pipeline stage 5: plate OCR and challan writing, which can take far longer than one video frame."""
        while self._running or not self._violation_queue.empty():
            try:
                tid, crop = self._violation_queue.get(timeout=PIPELINE_WAIT_SECONDS)
            except queue.Empty:
                continue
            try:
                plate_text, _plate_conf = self.plate_extractor.read_plate(crop)
                self.record_violation(tid, plate_text, crop)
            except Exception as exc:  # keep the live stream alive if OCR/DB fails
                print(f"[WARN] Failed to record violation for track #{tid}: {exc}")
