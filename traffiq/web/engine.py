"""
TRAFFIQ Web Dashboard Streaming & Computer Vision Engine.
Wraps vehicle detection, tracking, stop-line violation enforcement, OCR,
and dual-model ambulance priority into a thread-safe streaming engine.
"""

from __future__ import annotations

import collections
import os
import sys
import threading
import time
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

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
        self._thread: Optional[threading.Thread] = None
        self._latest_jpeg: Optional[bytes] = None

        # Statistics & Events (Bounded)
        self.stats: Dict[str, Any] = {
            "vehicles": 0,
            "violations": 0,
            "emergencies": 0,
            "total_fines": 0,
            "fps": 0.0,
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

        ret, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ret:
            with self._lock:
                self._latest_jpeg = buf.tobytes()

    def start(self) -> None:
        """Start the background vision loop thread."""
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._run_loop, daemon=True, name="DashboardEngineThread")
            self._thread.start()

    def stop(self) -> None:
        """Stop background execution."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

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
            return {
                "stats": {
                    "vehicles": int(self.stats["vehicles"]),
                    "violations": int(self.stats["violations"]),
                    "emergencies": int(self.stats["emergencies"]),
                    "total_fines": int(self.stats["total_fines"]),
                    "fps": round(float(self.stats["fps"]), 1),
                },
                "signal": {
                    "state": "GREEN" if self.is_emergency_locked else self.signal.state,
                    "time_remaining": round(float(rem), 1),
                    "is_emergency_locked": bool(self.is_emergency_locked),
                },
                "emergency_active": bool(self.is_emergency_locked),
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

    def _run_loop(self) -> None:
        """Main processing thread loop."""
        # 1. Initialize YOLO Models
        try:
            if os.path.exists(self.model_path):
                self.tracker = VehicleTracker(
                    model_path=self.model_path,
                    confidence_threshold=self.confidence_threshold,
                    device=self.device_str,
                )
            if os.path.exists(self.oiv7_path):
                self.oiv7_model = YOLO(self.oiv7_path)
                self.oiv7_model.to(self.device)
        except Exception as e:
            print(f"[WARN] Error loading models in DashboardEngine: {e}")

        # 2. Open Video Stream
        cap = None
        is_video_file = False
        src_str = str(self.source).strip("'\"")

        try:
            if not src_str.isdigit() and os.path.exists(src_str):
                is_video_file = True
                cap = cv2.VideoCapture(src_str)
            else:
                cam_idx = int(src_str) if src_str.isdigit() else 0
                cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
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
        except Exception as e:
            print(f"[WARN] Failed opening video source: {e}")

        tripwire = None
        frame_idx = 0
        prev_time = time.time()
        cached_amb_boxes: List[Tuple[np.ndarray, float]] = []

        while self._running:
            loop_start = time.time()

            # Handle missing or failed camera
            if cap is None or not cap.isOpened():
                # Produce synthetic preview and wait briefly
                self._create_placeholder_frame("No Camera Detected - Interactive Demo Ready")
                time.sleep(0.05)
                # Auto-demo check
                if self.auto_demo and (time.time() - self._last_event_time > 60.0):
                    self.trigger_demo_violation()
                continue

            ret, frame = cap.read()
            if not ret or frame is None:
                if is_video_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.03)
                    continue

            # Standardize resolution
            if frame.shape[1] > 1280:
                scale = 1280.0 / frame.shape[1]
                frame = cv2.resize(frame, (1280, int(frame.shape[0] * scale)), interpolation=cv2.INTER_LINEAR)

            h, w = frame.shape[:2]
            stop_y = int(h * self.tripwire_ratio)
            exit_y = int(h * EMERGENCY_EXIT_LINE_RATIO)

            if tripwire is None:
                tripwire = VirtualTripwire(y_position=stop_y)

            frame_idx += 1
            now = time.time()
            fps = 1.0 / max(1e-4, now - prev_time)
            prev_time = now

            with self._lock:
                self.stats["fps"] = fps

            # -----------------------------------------------------------------
            # Traffic Signal & Emergency Status Check
            # -----------------------------------------------------------------
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

            sig_state = "GREEN" if self.is_emergency_locked else self.signal.state

            # -----------------------------------------------------------------
            # Computer Vision Detection & Tracking
            # -----------------------------------------------------------------
            tracked_vehicles: List[TrackedVehicle] = []
            if self.tracker is not None:
                try:
                    tracked_vehicles = self.tracker.track(frame, frame_idx=frame_idx)
                except Exception:
                    tracked_vehicles = []

            # Ambulance detection (every 2 frames)
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

            # Process Tracked Vehicles & Violations
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
                    for abox, _amb_conf in cached_amb_boxes:
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
                    # Red light violation!
                    viol = self.violation_detector.evaluate_crossing(
                        track_id=tid,
                        crossed=True,
                        signal_state="RED",
                        bbox=bbox,
                        frame=frame,
                    )
                    if viol:
                        try:
                            plate_text, _plate_conf = self.plate_extractor.read_plate(viol.vehicle_crop)
                            self.record_violation(tid, plate_text, viol.vehicle_crop)
                        except Exception as exc:  # keep the live stream alive if OCR/DB fails
                            print(f"[WARN] Failed to record violation for track #{tid}: {exc}")

            # -----------------------------------------------------------------
            # Draw Overlays on Frame
            # -----------------------------------------------------------------
            # Stop line
            line_col = (0, 0, 255) if sig_state == "RED" else (0, 255, 0)
            cv2.line(frame, (0, stop_y), (w, stop_y), line_col, 3)
            cv2.putText(frame, f"STOP LINE [Signal: {sig_state}]", (20, stop_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, line_col, 2)

            # Bounding boxes
            for tv in tracked_vehicles:
                tid = int(tv.track_id)
                bx1, by1, bx2, by2 = tv.bbox
                is_amb = tid in self.active_emergency_tracks
                b_col = (0, 0, 255) if is_amb else (34, 197, 94)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), b_col, 2)
                lbl = f"AMBULANCE #{tid}" if is_amb else f"{tv.class_name} #{tid}"
                cv2.putText(frame, lbl, (bx1, max(15, by1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, b_col, 1)

            # Telemetry Pill in Top Left
            cv2.rectangle(frame, (10, 10), (320, 80), (15, 23, 42), -1)
            cv2.rectangle(frame, (10, 10), (320, 80), line_col, 1)
            cv2.putText(frame, f"SIGNAL: {sig_state}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, line_col, 2)
            cv2.putText(frame, f"FPS: {fps:.1f} | Vehicles: {self.stats['vehicles']}", (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

            # Emergency overlay banner
            if self.is_emergency_locked:
                cv2.rectangle(frame, (0, h - 55), (w, h), (22, 101, 52), -1)
                cv2.putText(frame, "EMERGENCY GREEN CORRIDOR ACTIVE", (int(w * 0.28), h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            # Encode to JPEG
            ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                with self._lock:
                    self._latest_jpeg = buf.tobytes()

            # Auto-demo check (if no events for 60s)
            if self.auto_demo and (time.time() - self._last_event_time > 60.0):
                self.trigger_demo_violation()

            # Rate limit loop: sleep only if frame completed ahead of 30fps target
            elapsed_loop = time.time() - loop_start
            sleep_dur = 0.033 - elapsed_loop
            if sleep_dur > 0:
                time.sleep(sleep_dur)

        if cap is not None:
            cap.release()
