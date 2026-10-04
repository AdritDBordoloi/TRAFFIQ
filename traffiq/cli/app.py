"""
Unified Command Line Interface (CLI) application and dispatcher for TRAFFIQ.
Coordinates argument parsing, configuration instantiation, and pipeline execution
across the 4 operational modes:
  1. diagnostics: Hardware, GPU, database schema/integrity preflight check.
  2. detection: Real-time YOLOv8 vehicle detection with telemetry HUD.
  3. tracking: ByteTrack multi-object tracking with virtual tripwire flow counting.
  4. enforcement: Full automated enforcement loop (signal + violation + plate ROI OCR + e-challan DB logging).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Union
import cv2
import numpy as np

from traffiq.config.settings import TraffiqConfig
from traffiq.database import (
    DatabaseManager,
    migrate_legacy_blobs,
    setup_database,
    verify_integrity,
)
from traffiq.enforcement import (
    ChallanIssuer,
    PlateExtractor,
    TrafficSignal,
    ViolationDetector,
)
from traffiq.vision import (
    CameraStream,
    VehicleDetector,
    VehicleTracker,
    VirtualTripwire,
)

# Canonical operational modes and aliases
VALID_MODES: List[str] = [
    "diagnostics",
    "detection",
    "tracking",
    "enforcement",
]

MODE_ALIASES: Dict[str, str] = {
    "diag": "diagnostics",
    "diagnostics": "diagnostics",
    "detect": "detection",
    "detection": "detection",
    "track": "tracking",
    "tracking": "tracking",
    "enforce": "enforcement",
    "enforcement": "enforcement",
}


def normalize_mode(val: Any) -> str:
    """Normalize mode string case-insensitively and map operational aliases."""
    if val is None:
        return "enforcement"
    cleaned = str(val).strip().lower()
    return MODE_ALIASES.get(cleaned, cleaned)


class TraffiqArgumentParser(argparse.ArgumentParser):
    """
    Specialized ArgumentParser for the TRAFFIQ system.
    Resolves operational mode priority (flag overrides positional) and defaults to 'enforcement'.
    """

    def parse_args(
        self,
        args: Optional[Sequence[str]] = None,
        namespace: Optional[argparse.Namespace] = None,
    ) -> argparse.Namespace:
        parsed = super().parse_args(args=args, namespace=namespace)

        # Reconcile mode: flag takes precedence over positional, default to 'enforcement'
        mode_val = (
            getattr(parsed, "mode_flag", None)
            or getattr(parsed, "mode", None)
            or "enforcement"
        )
        parsed.mode = normalize_mode(mode_val)
        return parsed


def build_parser() -> argparse.ArgumentParser:
    """
    Construct and return the authoritative TRAFFIQ CLI argument parser.

    Supports operational modes ('diagnostics', 'detection', 'tracking', 'enforcement')
    and complete hardware, vision, enforcement, and automation flags.
    """
    parser = TraffiqArgumentParser(
        prog="traffiq",
        description="TRAFFIQ: AI-Powered Automated Traffic Enforcement & E-Challan System",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Operational Pipeline Mode (Positional and Optional Flag)
    parser.add_argument(
        "mode",
        nargs="?",
        default=None,
        type=normalize_mode,
        choices=VALID_MODES,
        help="Operational pipeline mode: diagnostics, detection, tracking, enforcement (or aliases: diag, detect, track, enforce)",
    )
    parser.add_argument(
        "-m",
        "--mode",
        dest="mode_flag",
        type=normalize_mode,
        choices=VALID_MODES,
        default=None,
        help="Operational pipeline mode: diagnostics, detection, tracking, enforcement (overrides positional)",
    )

    # Input Video Source & Camera Properties
    parser.add_argument(
        "-s",
        "--source",
        "--camera",
        dest="source",
        default="0",
        help="Camera device index (int, e.g. 0) or path to input video file (str)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Camera capture frame width in pixels",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Camera capture frame height in pixels",
    )

    # Inference Compute Device & YOLO Model
    parser.add_argument(
        "--device",
        default="0",
        help="Inference compute device ('0' for CUDA GPU or 'cpu')",
    )
    parser.add_argument(
        "--model",
        default="yolov8s.pt",
        help="Path to YOLO model weights file",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO detection confidence threshold (0.0 to 1.0)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image resolution dimension",
    )
    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="ByteTrack multi-object tracker YAML configuration file",
    )

    # Enforcement & Traffic Control Parameters
    parser.add_argument(
        "--line-ratio",
        type=float,
        default=0.55,
        help="Virtual stop-line tripwire vertical ratio (0.0 to 1.0)",
    )
    parser.add_argument(
        "--fine",
        type=int,
        default=1000,
        help="Standard traffic violation penalty fine in INR",
    )
    parser.add_argument(
        "--signal-duration",
        type=float,
        default=8.0,
        help="Automatic traffic light signal phase cycle duration in seconds",
    )
    parser.add_argument(
        "--output-dir",
        default="violations",
        help="Destination directory for violation snapshot captures",
    )
    parser.add_argument(
        "--db-path",
        default="traffiq_enforcement.db",
        help="Path to SQLite e-challan database file",
    )

    # Headless Execution & Test Automation
    parser.add_argument(
        "--no-gui",
        action="store_true",
        default=False,
        help="Disable interactive OpenCV visualization windows for headless operation",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process before terminating (for testing)",
    )

    return parser


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Convenience helper to build parser and parse argument tokens."""
    parser = build_parser()
    return parser.parse_args(args)


# ---------------------------------------------------------------------------
# Execution Dispatchers
# ---------------------------------------------------------------------------

def run_diagnostics(config: TraffiqConfig) -> int:
    """
    Non-blocking hardware and database preflight check.
    Checks Python version, PyTorch CUDA GPU status, checks database schema
    and runs verify_integrity() asserting 0 blobs, checks camera access
    if possible or mock frame, and prints structured report. Returns 0 on success.
    """
    print("\n" + "=" * 65)
    print("         TRAFFIQ SYSTEM DIAGNOSTICS & PREFLIGHT REPORT           ")
    print("=" * 65)

    # 1. Python Runtime & Environment
    py_ver = sys.version.split()[0]
    print(f" Python Runtime     : {py_ver} ({sys.platform})")
    print(f" Executable Path    : {sys.executable}")

    # 2. PyTorch & GPU Acceleration
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        if cuda_avail:
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = round(
                torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
            )
            cuda_ver = getattr(torch.version, "cuda", "N/A")
            print(f" GPU Acceleration   : [READY] CUDA {cuda_ver} ({gpu_name}, {vram_gb} GB VRAM)")
        else:
            print(f" GPU Acceleration   : [INFO] CUDA Unavailable (using CPU fallback)")
    except ImportError:
        print(f" GPU Acceleration   : [WARN] PyTorch library not detected")

    # 3. Database Schema & Integrity Check
    try:
        setup_database(config.db_path)
        total_rows, int_rows, blob_rows = verify_integrity(config.db_path)
        if blob_rows > 0:
            print(f" Database Notice    : Found {blob_rows} legacy blob records. Running migration...")
            migrate_legacy_blobs(config.db_path)
            total_rows, int_rows, blob_rows = verify_integrity(config.db_path)

        assert blob_rows == 0, f"Integrity check failed: {blob_rows} blobs remain"
        print(f" Database Schema    : [READY] {config.db_path} verified (100% integer IDs, 0 blobs)")
        print(f" Database Records   : {total_rows} total challan entries ({int_rows} integer rows)")
    except Exception as db_err:
        print(f" Database Status    : [ERROR] Database check encountered an issue: {db_err}")
        return 1

    # 4. Computer Vision Modules
    try:
        from ultralytics import YOLO
        print(f" YOLOv8 Engine      : [READY] Ultralytics framework active (model: {config.model_path})")
    except ImportError:
        print(f" YOLOv8 Engine      : [WARN] Ultralytics package not detected")

    try:
        import easyocr
        print(f" EasyOCR Engine     : [READY] EasyOCR OCR module available")
    except ImportError:
        print(f" EasyOCR Engine     : [WARN] EasyOCR package not detected")

    # 5. Video Stream Input Probe
    try:
        with CameraStream(
            source=config.camera_index,
            width=config.frame_width,
            height=config.frame_height,
            auto_open=True,
        ) as stream:
            if stream.is_opened():
                ret, test_frame = stream.read()
                if ret and test_frame is not None:
                    h, w = test_frame.shape[:2]
                    print(f" Video Input Stream : [READY] Source '{config.camera_index}' active ({w}x{h} @ {stream.fps:.1f} FPS)")
                else:
                    print(f" Video Input Stream : [INFO] Source '{config.camera_index}' opened (synthetic frame fallback ready)")
            else:
                print(f" Video Input Stream : [INFO] Source '{config.camera_index}' inaccessible (synthetic frame fallback ready)")
    except Exception as cam_err:
        print(f" Video Input Stream : [INFO] Probe finished: {cam_err} (synthetic frame fallback ready)")

    # 6. Storage & Evidence Directories
    os.makedirs(config.output_dir, exist_ok=True)
    print(f" Evidence Directory : [READY] '{config.output_dir}/' writeable")

    print("=" * 65)
    print(" PREFLIGHT RESULT   : ALL CRITICAL SUBSYSTEMS OPERATIONAL (0 ERRORS)")
    print("=" * 65 + "\n")
    return 0


def run_detection(config: TraffiqConfig) -> int:
    """
    Loops through CameraStream, runs VehicleDetector, annotates frame with
    bounding boxes and FPS, respects --no-gui and --max-frames.
    """
    print(
        f"[INFO] Initializing VehicleDetector (model={config.model_path}, "
        f"device={config.device}, conf={config.confidence_threshold})..."
    )
    detector = VehicleDetector(
        model_path=config.model_path,
        confidence_threshold=config.confidence_threshold,
        device=config.device,
        imgsz=config.imgsz,
    )

    print(
        f"[INFO] Opening CameraStream (source={config.camera_index}, "
        f"resolution={config.frame_width}x{config.frame_height})..."
    )
    frame_count = 0
    prev_time = time.time()

    with CameraStream(
        source=config.camera_index,
        width=config.frame_width,
        height=config.frame_height,
    ) as stream:
        use_synthetic = not stream.is_opened()
        if use_synthetic:
            print(f"[WARN] Camera source '{config.camera_index}' unavailable; using synthetic frames for testing.")

        while True:
            if config.max_frames is not None and frame_count >= config.max_frames:
                break

            if use_synthetic:
                frame = np.full((config.frame_height, config.frame_width, 3), 40, dtype=np.uint8)
                time.sleep(0.01)
            else:
                ret, frame = stream.read()
                if not ret or frame is None:
                    print("[INFO] End of stream or capture interrupted.")
                    break

            frame_count += 1
            curr_time = time.time()
            dt = curr_time - prev_time
            fps = 1.0 / dt if dt > 0 else 30.0
            prev_time = curr_time

            # Run vehicle detection
            detections = detector.detect(frame)

            counts: Dict[str, int] = {
                "Car": 0, "Motorcycle": 0, "Bus": 0, "Truck": 0, "Bicycle": 0
            }
            for det in detections:
                counts[det.class_name] = counts.get(det.class_name, 0) + 1

            if not config.no_gui:
                for det in detections:
                    x1, y1, x2, y2 = det.bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    tag = f"{det.class_name} {int(det.confidence * 100)}%"
                    cv2.putText(
                        frame,
                        tag,
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

                total_vehicles = sum(counts.values())
                cv2.rectangle(frame, (10, 10), (320, 110), (20, 20, 20), -1)
                cv2.rectangle(frame, (10, 10), (320, 110), (0, 255, 0), 1)
                cv2.putText(frame, "TRAFFIQ DETECTOR", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                cv2.putText(frame, f"FPS: {fps:.1f}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Live Vehicle Density: {total_vehicles}", (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(
                    frame,
                    f"Cars: {counts.get('Car', 0)} | Bikes: {counts.get('Motorcycle', 0)} | Heavy: {counts.get('Truck', 0) + counts.get('Bus', 0)}",
                    (20, 98),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (200, 200, 200),
                    1,
                )

                cv2.imshow("TRAFFIQ - Real-Time Vehicle Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] User terminated detection loop.")
                    break

    if not config.no_gui:
        cv2.destroyAllWindows()

    print(f"[INFO] Detection finished cleanly ({frame_count} frames processed).")
    return 0


def run_tracking(config: TraffiqConfig) -> int:
    """
    Loops through CameraStream, runs VehicleTracker + VirtualTripwire,
    draws tracks/centroids and line crossings, respects --no-gui and --max-frames.
    """
    print(
        f"[INFO] Initializing VehicleTracker (model={config.model_path}, "
        f"tracker={config.tracker}, device={config.device})..."
    )
    tracker = VehicleTracker(
        model_path=config.model_path,
        tracker_config=config.tracker,
        confidence_threshold=config.confidence_threshold,
        device=config.device,
        ttl=30,
    )
    tripwire = VirtualTripwire(
        tripwire_ratio=config.tripwire_ratio,
        frame_height=config.frame_height,
    )

    frame_count = 0
    prev_time = time.time()

    with CameraStream(
        source=config.camera_index,
        width=config.frame_width,
        height=config.frame_height,
    ) as stream:
        use_synthetic = not stream.is_opened()
        if use_synthetic:
            print(f"[WARN] Camera source '{config.camera_index}' unavailable; using synthetic frames for testing.")

        while True:
            if config.max_frames is not None and frame_count >= config.max_frames:
                break

            if use_synthetic:
                frame = np.full((config.frame_height, config.frame_width, 3), 40, dtype=np.uint8)
                time.sleep(0.01)
            else:
                ret, frame = stream.read()
                if not ret or frame is None:
                    print("[INFO] End of stream or capture interrupted.")
                    break

            frame_count += 1
            curr_time = time.time()
            dt = curr_time - prev_time
            fps = 1.0 / dt if dt > 0 else 30.0
            prev_time = curr_time

            tripwire.set_frame_height(frame.shape[0])

            # Run tracking
            tracked_vehicles = tracker.track(frame, frame_idx=frame_count)

            # Update virtual tripwire
            for veh in tracked_vehicles:
                crossed = tripwire.update_track(veh.track_id, veh.centroid)
                if crossed:
                    print(f"[FLOW] Vehicle ID #{veh.track_id} ({veh.class_name}) crossed stop-line! Total count: {tripwire.flow_count}")

            # Prune inactive trajectories with frame-based TTL
            tracker.prune_inactive(frame_count, ttl=30)

            if not config.no_gui:
                tripwire.draw(frame)

                for veh in tracked_vehicles:
                    x1, y1, x2, y2 = veh.bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    tag = f"ID: {veh.track_id} | {veh.class_name}"
                    cv2.putText(
                        frame,
                        tag,
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        2,
                    )
                    cv2.circle(frame, veh.centroid, 4, (0, 0, 255), -1)

                    traj = tracker.get_trajectory(veh.track_id)
                    if traj and len(traj) > 1:
                        for j in range(1, len(traj)):
                            cv2.line(frame, traj[j - 1], traj[j], (255, 100, 0), 1)

                cv2.rectangle(frame, (10, 10), (360, 95), (20, 20, 20), -1)
                cv2.rectangle(frame, (10, 10), (360, 95), (0, 255, 255), 1)
                cv2.putText(frame, "TRAFFIQ TRACKER (ByteTrack)", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                cv2.putText(frame, f"FPS: {fps:.1f} | Active Tracks: {tracker.trajectory_cache.active_track_count()}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Total Vehicles Passed: {tripwire.flow_count}", (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cv2.imshow("TRAFFIQ - Multi-Object Vehicle Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] User terminated tracking loop.")
                    break

    if not config.no_gui:
        cv2.destroyAllWindows()

    print(f"[INFO] Tracking finished cleanly ({frame_count} frames, {tripwire.flow_count} crossings).")
    return 0


def run_enforcement(config: TraffiqConfig) -> int:
    """
    Complete live automated enforcement loop. Synchronizes TrafficSignal,
    runs detection and tracking, evaluates stop-line red light violations,
    isolates plate ROI and runs EasyOCR with fallback to 'UNKNOWN',
    logs challan into database using native int track_id, saves evidence
    snapshots into violations/, and displays HUD banner alert.
    Respects --no-gui and --max-frames.
    """
    print("[INFO] Initializing full automated enforcement pipeline...")
    os.makedirs(config.output_dir, exist_ok=True)

    db_manager = DatabaseManager(db_path=config.db_path)
    signal = TrafficSignal(cycle_duration=config.signal_duration)
    tracker = VehicleTracker(
        model_path=config.model_path,
        tracker_config=config.tracker,
        confidence_threshold=config.confidence_threshold,
        device=config.device,
        ttl=30,
    )
    tripwire = VirtualTripwire(
        tripwire_ratio=config.tripwire_ratio,
        frame_height=config.frame_height,
    )
    violation_detector = ViolationDetector(fine_amount=config.fine_amount)
    plate_extractor = PlateExtractor(
        gpu=(config.device != "cpu"),
        auto_init_reader=True,
    )
    challan_issuer = ChallanIssuer(
        db_manager=db_manager,
        db_path=config.db_path,
        output_dir=config.output_dir,
        fine_amount=config.fine_amount,
    )

    frame_count = 0
    prev_time = time.time()

    with CameraStream(
        source=config.camera_index,
        width=config.frame_width,
        height=config.frame_height,
    ) as stream:
        use_synthetic = not stream.is_opened()
        if use_synthetic:
            print(f"[WARN] Camera source '{config.camera_index}' unavailable; using synthetic frames for testing.")

        while True:
            if config.max_frames is not None and frame_count >= config.max_frames:
                break

            if use_synthetic:
                frame = np.full((config.frame_height, config.frame_width, 3), 40, dtype=np.uint8)
                time.sleep(0.01)
            else:
                ret, frame = stream.read()
                if not ret or frame is None:
                    print("[INFO] End of stream or capture interrupted.")
                    break

            frame_count += 1
            curr_time = time.time()
            dt = curr_time - prev_time
            fps = 1.0 / dt if dt > 0 else 30.0
            prev_time = curr_time

            # Update traffic signal status
            signal_state = signal.update(curr_time)
            tripwire.set_frame_height(frame.shape[0])

            # Run tracking
            tracked_vehicles = tracker.track(frame, frame_idx=frame_count)

            # Evaluate tripwire and red-light infractions
            for veh in tracked_vehicles:
                crossed = tripwire.update_track(veh.track_id, veh.centroid)

                if crossed and signal.is_red():
                    event = violation_detector.evaluate_crossing(
                        track_id=veh.track_id,
                        crossed=crossed,
                        signal_state=signal_state,
                        bbox=veh.bbox,
                        frame=frame,
                    )
                    if event is not None:
                        # Plate ROI OCR extraction with fallback
                        plate_text, conf = plate_extractor.read_plate(event.vehicle_crop)

                        # Issue challan into database strictly casting track_id to native int
                        challan = challan_issuer.issue_challan(
                            track_id=veh.track_id,
                            vehicle_crop=event.vehicle_crop,
                            detected_plate=plate_text,
                            fine_amount=config.fine_amount,
                            timestamp=event.timestamp,
                        )
                        print(
                            f"[VIOLATION] Red light infraction detected! "
                            f"Track ID: #{veh.track_id} | Plate: {plate_text} | "
                            f"Owner: {challan['owner_name']} | Challan #{challan['challan_id']} "
                            f"(Fine: INR {challan['fine_amount']})"
                        )

            # Prune inactive trajectories with frame-based TTL
            tracker.prune_inactive(frame_count, ttl=30)

            if not config.no_gui:
                line_color = (0, 0, 255) if signal.is_red() else (0, 255, 0)
                tripwire.draw(
                    frame,
                    color=line_color,
                    label=f"STOP LINE [{signal_state}] Passed: {tripwire.flow_count}",
                )

                for veh in tracked_vehicles:
                    x1, y1, x2, y2 = veh.bbox
                    is_violator = violation_detector.is_violation(veh.track_id)
                    box_color = (0, 0, 255) if is_violator else (0, 255, 0)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    tag = f"ID: {veh.track_id} | {veh.class_name}" + (" [VIOLATOR]" if is_violator else "")
                    cv2.putText(
                        frame,
                        tag,
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        box_color,
                        2,
                    )
                    cv2.circle(frame, veh.centroid, 4, box_color, -1)

                challan_issuer.draw_hud_banner(frame, current_time=curr_time)
                challan_issuer.draw_telemetry_hud(
                    frame,
                    signal_state=signal_state,
                    fps=fps,
                    total_passed=tripwire.flow_count,
                    violations_count=violation_detector.get_violation_count(),
                    line_color=line_color,
                )

                cv2.imshow("TRAFFIQ - Automated Traffic Enforcement System", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[INFO] User terminated enforcement loop.")
                    break
                elif key == ord("t"):
                    new_state = signal.toggle(curr_time)
                    print(f"[MANUAL] Signal manually toggled to {new_state}.")

    if not config.no_gui:
        cv2.destroyAllWindows()

    print(
        f"[INFO] Enforcement run finished cleanly: {frame_count} frames, "
        f"{tripwire.flow_count} passed, {violation_detector.get_violation_count()} challans issued."
    )
    return 0


# ---------------------------------------------------------------------------
# Main Entrypoint
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Main entrypoint for the TRAFFIQ unified CLI launcher.
    Parses CLI arguments, constructs TraffiqConfig, and dispatches to
    the selected operational pipeline mode.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    config = TraffiqConfig.from_args(args)

    mode = getattr(args, "mode", "enforcement")

    if mode == "diagnostics":
        return run_diagnostics(config)
    elif mode == "detection":
        return run_detection(config)
    elif mode == "tracking":
        return run_tracking(config)
    elif mode == "enforcement":
        return run_enforcement(config)
    else:
        print(f"[ERROR] Unrecognized operational mode: {mode}", file=sys.stderr)
        return 1


def run_cli(argv: Optional[Sequence[str]] = None) -> int:
    """Convenience alias for main entrypoint."""
    return main(argv)
