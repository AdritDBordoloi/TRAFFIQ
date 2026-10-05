# live_priority.py
import cv2
import time
import os
import sys
import sqlite3
import winsound
from datetime import datetime
from ultralytics import YOLO

def init_emergency_db():
    conn = sqlite3.connect("traffiq_enforcement.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS emergency_events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        track_id INT NOT NULL,
        vehicle_type TEXT NOT NULL,
        override_duration_sec REAL NOT NULL,
        status TEXT NOT NULL
    )
    """)
    conn.commit()
    conn.close()

def log_emergency_event(track_id, vehicle_type, duration, status):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect("traffiq_enforcement.db")
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO emergency_events (timestamp, track_id, vehicle_type, override_duration_sec, status)
    VALUES (?, ?, ?, ?, ?)
    """, (timestamp, track_id, vehicle_type, round(duration, 2), status))
    conn.commit()
    conn.close()
    print("\n" + "=" * 65)
    print(f" [!] GREEN CORRIDOR RESTORED: Track #{track_id} ({vehicle_type})")
    print(f"     Duration: {duration:.1f}s | Resolution: {status} | DB Saved")
    print("=" * 65 + "\n")

def run_priority_system(camera_index=0):
    init_emergency_db()

    print("[INFO] Loading YOLOv8s Open Images V7 on RTX 4060 GPU...")
    model = YOLO("yolov8s-oiv7.pt")

    # Map target vehicle and ambulance classes
    TARGET_CLASSES = {}
    ambulance_ids = set()

    for cls_id, name in model.names.items():
        name_lower = name.lower()
        if "ambulance" in name_lower:
            TARGET_CLASSES[cls_id] = "Ambulance"
            ambulance_ids.add(cls_id)
        elif name_lower in ["car", "bus", "truck", "motorcycle", "van"]:
            TARGET_CLASSES[cls_id] = name.capitalize()

    print(f"[INFO] Monitoring classes: {list(set(TARGET_CLASSES.values()))}")
    print(f"[INFO] Ambulance Model Class ID(s): {ambulance_ids}")

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {camera_index}.")
        return

    # Tracking & State Management
    track_positions = {}
    active_emergency_track = None
    override_start_time = None
    last_seen_ambulance_time = 0
    safety_timeout = 25.0  # Max override window in seconds

    # Traffic signal cycle
    light_state = "GREEN"
    last_light_switch = time.time()
    normal_cycle_duration = 80000.0

    prev_time = time.time()
    print("\n[SUCCESS] TRAFFIQ Automatic Priority Engine Active.")
    print("[INFO] Show an ambulance to automatically trigger the Green Corridor.")
    print("[CONTROLS] Press 't' to toggle light manually | 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        stop_line_y = int(height * 0.50)   # Intersection entry stop line
        exit_line_y = int(height * 0.72)   # Intersection exit clearance line

        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time

        # Normal Signal Rotation (Only active when NO ambulance is overriding)
        if active_emergency_track is None:
            if current_time - last_light_switch > normal_cycle_duration:
                light_state = "RED" if light_state == "GREEN" else "GREEN"
                last_light_switch = current_time
        else:
            # Emergency override actively forces GREEN
            light_state = "GREEN"

            # Failsafe 1: Max duration ceiling
            elapsed_override = current_time - override_start_time
            if elapsed_override > safety_timeout:
                log_emergency_event(active_emergency_track, "Ambulance", elapsed_override, "TIMEOUT_CLEARED")
                active_emergency_track = None
                override_start_time = None
                last_light_switch = current_time

            # Failsafe 2: Ambulance vanished from scene for over 4 seconds
            elif current_time - last_seen_ambulance_time > 4.0:
                log_emergency_event(active_emergency_track, "Ambulance", elapsed_override, "LOST_TRACK_CLEARED")
                active_emergency_track = None
                override_start_time = None
                last_light_switch = current_time

        # Run ByteTrack Tracking on GPU
        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=list(TARGET_CLASSES.keys()),
            conf=0.25,
            imgsz=640,
            device=0,
            verbose=False
        )[0]

        ambulance_detected_in_frame = False

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            confidences = results.boxes.conf.cpu().numpy()

            for box, track_id, cls_id, conf in zip(boxes, track_ids, class_ids, confidences):
                x1, y1, x2, y2 = box
                cls_name = TARGET_CLASSES.get(cls_id, "Vehicle")
                is_ambulance = cls_id in ambulance_ids
                cy = int((y1 + y2) / 2)
                cx = int((x1 + x2) / 2)

                if track_id not in track_positions:
                    track_positions[track_id] = cy
                prev_cy = track_positions[track_id]
                track_positions[track_id] = cy

                # ========================================================
                # AUTOMATIC AMBULANCE OVERRIDE LOGIC
                # ========================================================
                if is_ambulance:
                    ambulance_detected_in_frame = True
                    last_seen_ambulance_time = current_time

                    # Automatically engage override if not already locked
                    if active_emergency_track is None and cy < exit_line_y:
                        active_emergency_track = track_id
                        override_start_time = current_time
                        print("\n" + "=" * 65)
                        print(f" [!!!] AUTOMATIC GREEN CORRIDOR ENGAGED: Track ID #{track_id}")
                        print(f"       Class: Ambulance ({int(conf * 100)}% conf) | Forcing Signal: GREEN")
                        print("=" * 65)
                        # Audible alert on Windows
                        try:
                            winsound.Beep(1500, 350)
                        except Exception:
                            pass

                # Check if the active emergency vehicle crossed the Exit Line
                if track_id == active_emergency_track:
                    if prev_cy < exit_line_y <= cy:
                        elapsed = current_time - override_start_time
                        log_emergency_event(track_id, "Ambulance", elapsed, "CLEARED_EXIT")
                        active_emergency_track = None
                        override_start_time = None
                        last_light_switch = current_time

                # Bounding Box Visualization
                if is_ambulance or (track_id == active_emergency_track):
                    box_color = (0, 0, 255)  # Bright Red for Ambulance
                    label = f"EMERGENCY: {cls_name} #{track_id} {int(conf * 100)}%"
                    box_thickness = 3
                else:
                    box_color = (0, 255, 0) if light_state == "GREEN" else (0, 165, 255)
                    label = f"ID:{track_id} {cls_name} {int(conf * 100)}%"
                    box_thickness = 2

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, label, (x1, max(20, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 2)

        # Draw Intersection Boundaries
        stop_color = (0, 255, 0) if light_state == "GREEN" else (0, 0, 255)
        cv2.line(frame, (0, stop_line_y), (width, stop_line_y), stop_color, 2)
        cv2.putText(frame, f"STOP LINE [{light_state}]", (10, stop_line_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, stop_color, 2)

        cv2.line(frame, (0, exit_line_y), (width, exit_line_y), (255, 255, 0), 2)
        cv2.putText(frame, "CLEARANCE EXIT LINE", (10, exit_line_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Flashing Strobe Emergency HUD Banner (Red & Blue alternating)
        if active_emergency_track is not None:
            strobe_color = (0, 0, 200) if int(time.time() * 4) % 2 == 0 else (200, 50, 0)
            cv2.rectangle(frame, (20, height - 90), (width - 20, height - 20), strobe_color, -1)
            cv2.rectangle(frame, (20, height - 90), (width - 20, height - 20), (255, 255, 255), 2)
            cv2.putText(frame, "EMERGENCY VEHICLE DETECTED - GREEN CORRIDOR ACTIVE", 
                        (40, height - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)
            elapsed_txt = f"Signal Held GREEN | Elapsed: {time.time() - override_start_time:.1f}s | Priority Lock: Active"
            cv2.putText(frame, elapsed_txt, (40, height - 32), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 255, 255), 1)

        # Upper Telemetry HUD
        hud_border = (0, 0, 255) if active_emergency_track else (0, 255, 0)
        cv2.rectangle(frame, (10, 10), (410, 115), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (410, 115), hud_border, 2)

        mode_txt = "EMERGENCY OVERRIDE" if active_emergency_track else "NORMAL CYCLE"
        cv2.putText(frame, f"MODE: {mode_txt}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, stop_color, 2)
        cv2.putText(frame, f"SIGNAL: {light_state}", (270, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, stop_color, 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Model: YOLOv8s-OIV7 (Native Ambulance)", (20, 88),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        cv2.putText(frame, "[Controls: 't' Toggle Light | 'q' Quit]", (20, 108),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

        cv2.imshow("TRAFFIQ - Automated Priority Green Corridor", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            if active_emergency_track is None:
                light_state = "RED" if light_state == "GREEN" else "GREEN"
                last_light_switch = time.time()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_priority_system(0)