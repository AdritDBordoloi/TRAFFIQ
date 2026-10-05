import cv2
import time
import os
import sys
import sqlite3
from datetime import datetime
from ultralytics import YOLO

TRAFFIC_CLASSES = {
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

def get_vehicle_semi_crop(frame, box, scale=2.0, min_w=320, min_h=240):
    """
    Extracts a semi close-up photo of the vehicle with the vehicle centered in the frame.
    """
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    crop_w = int(w * scale)
    crop_h = int(h * scale)
    crop_w = max(crop_w, min_w)
    crop_h = max(crop_h, min_h)

    if crop_w < int(crop_h * 1.2):
        crop_w = int(crop_h * 1.2)

    crop_w = min(width, crop_w)
    crop_h = min(height, crop_h)

    half_w = crop_w // 2
    half_h = crop_h // 2

    c_x1 = cx - half_w
    c_x2 = c_x1 + crop_w
    c_y1 = cy - half_h
    c_y2 = c_y1 + crop_h

    if c_x1 < 0:
        c_x2 = min(width, c_x2 - c_x1)
        c_x1 = 0
    elif c_x2 > width:
        c_x1 = max(0, c_x1 - (c_x2 - width))
        c_x2 = width

    if c_y1 < 0:
        c_y2 = min(height, c_y2 - c_y1)
        c_y1 = 0
    elif c_y2 > height:
        c_y1 = max(0, c_y1 - (c_y2 - height))
        c_y2 = height

    return frame[c_y1:c_y2, c_x1:c_x2]

def create_challan_notice(track_id, cls_name, evidence_path, vehicle_dir):
    """Generates an electronic challan notice receipt text file and logs to database if available."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fine_amount = 1000  # Fine amount in INR
    challan_id = int(time.time() * 1000) % 1000000

    # Log to SQLite database if available
    try:
        conn = sqlite3.connect("traffiq_enforcement.db")
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS challan_records (
            challan_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            track_id INT NOT NULL,
            plate_number TEXT NOT NULL,
            owner_name TEXT NOT NULL,
            contact_number TEXT NOT NULL,
            vehicle_model TEXT NOT NULL,
            violation_type TEXT NOT NULL,
            fine_amount INT NOT NULL,
            full_evidence_path TEXT NOT NULL,
            crop_evidence_path TEXT NOT NULL
        )
        """)
        cursor.execute("""
        INSERT INTO challan_records (
            timestamp, track_id, plate_number, owner_name, contact_number, 
            vehicle_model, violation_type, fine_amount, full_evidence_path, crop_evidence_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp, int(track_id), "PENDING_OCR", "Unverified Owner", "NOT ON FILE",
            cls_name, "Red Light Signal Violation", fine_amount,
            evidence_path, ""
        ))
        challan_id = cursor.lastrowid
        conn.commit()
        conn.close()
    except Exception:
        pass

    receipt_filename = f"{vehicle_dir}/challan.txt"
    with open(receipt_filename, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("       TRAFFIQ AUTOMATED TRAFFIC ENFORCEMENT NOTICE     \n")
        f.write("=" * 60 + "\n")
        f.write(f"Challan Number     : TRFQ-{challan_id:06d}\n")
        f.write(f"Violation DateTime : {timestamp}\n")
        f.write(f"Violation Type     : Red Light Disobedience (Section 119/177)\n")
        f.write(f"Vehicle Track ID   : #{track_id}\n")
        f.write(f"Vehicle Type       : {cls_name}\n")
        f.write(f"Fine Amount Due    : INR {fine_amount}\n")
        f.write(f"Vehicle Evidence   : {evidence_path}\n")
        f.write("=" * 60 + "\n")

    return receipt_filename, challan_id

def run_violation_detection(camera_index=0):
    # Ensure folder exists for saving snapshot evidence
    os.makedirs("violations", exist_ok=True)

    print("[INFO] Loading YOLOv8s with ByteTrack on RTX 4060...")
    model = YOLO("yolov8s.pt")

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {camera_index}.")
        return

    # Memory sets
    track_positions = {}
    counted_ids = set()
    violated_ids = set()

    # Telemetry metrics
    total_passed = 0
    violation_count = 0

    # Traffic light state simulation (Toggles automatically every 8s, or press 't')
    light_state = "GREEN"
    last_light_switch = time.time()
    light_cycle_duration = 8.0  # seconds

    prev_time = time.time()
    print("[SUCCESS] Violation detection engine active.")
    print("[CONTROLS] Press 't' to toggle Red/Green manually. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        stop_line_y = int(height * 0.55)

        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time

        # Automatic signal cycle
        if current_time - last_light_switch > light_cycle_duration:
            light_state = "RED" if light_state == "GREEN" else "GREEN"
            last_light_switch = current_time

        # Run ByteTrack tracking
        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=list(TRAFFIC_CLASSES.keys()),
            conf=0.25,
            imgsz=640,
            device=0,
            verbose=False
        )[0]

        # Draw stop line: RED if light is red, GREEN if light is green
        line_color = (0, 0, 255) if light_state == "RED" else (0, 255, 0)
        cv2.line(frame, (0, stop_line_y), (width, stop_line_y), line_color, 3)
        cv2.putText(
            frame, f"STOP LINE [{light_state}]", 
            (10, stop_line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, line_color, 2
        )

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            confidences = results.boxes.conf.cpu().numpy()

            for box, track_id, cls_id, conf in zip(boxes, track_ids, class_ids, confidences):
                x1, y1, x2, y2 = box
                cls_name = TRAFFIC_CLASSES.get(cls_id, "Vehicle")
                cy = int((y1 + y2) / 2)
                cx = int((x1 + x2) / 2)

                if track_id not in track_positions:
                    track_positions[track_id] = cy
                prev_cy = track_positions[track_id]
                track_positions[track_id] = cy

                # Check if centroid crossed the stop line from top to bottom
                if prev_cy < stop_line_y <= cy:
                    if track_id not in counted_ids:
                        counted_ids.add(track_id)
                        total_passed += 1

                    # Check for Red-Light Violation
                    if light_state == "RED" and track_id not in violated_ids:
                        violated_ids.add(track_id)
                        violation_count += 1

                        # Save evidence crop with timestamp in dedicated vehicle folder
                        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                        vehicle_dir = f"violations/vehicle_ID{track_id}_{timestamp_str}"
                        os.makedirs(vehicle_dir, exist_ok=True)

                        evidence_crop = get_vehicle_semi_crop(frame, (x1, y1, x2, y2))
                        filename = f"{vehicle_dir}/vehicle_image.jpg"
                        if evidence_crop.size > 0:
                            cv2.imwrite(filename, evidence_crop)
                        else:
                            cv2.imwrite(filename, frame)

                        # Generate challan notice inside vehicle folder
                        receipt_path, challan_id = create_challan_notice(track_id, cls_name, filename, vehicle_dir)
                        print(f"[ALERT] Red-Light Violation logged for ID:{track_id}! Challan: TRFQ-{challan_id:06d} | Receipt: {receipt_path}")

                # Bounding box rendering: Bright Red if violated, otherwise standard
                if track_id in violated_ids:
                    box_color = (0, 0, 255)
                    label = f"VIOLATION! ID:{track_id} {cls_name}"
                else:
                    box_color = (0, 255, 0) if track_id in counted_ids else (255, 150, 0)
                    label = f"ID:{track_id} {cls_name} {int(conf * 100)}%"

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, label, (x1, max(20, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 2)

        # Clear expired tracks
        if len(track_positions) > 500:
            track_positions.clear()

        # Telemetry Display
        cv2.rectangle(frame, (10, 10), (370, 140), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (370, 140), line_color, 2)

        # Signal indicator badge
        signal_badge_color = (0, 0, 255) if light_state == "RED" else (0, 255, 0)
        cv2.putText(frame, f"SIGNAL: {light_state}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, signal_badge_color, 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (200, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(frame, f"Flow Count (Total): {total_passed}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        cv2.putText(frame, f"Red-Light Violations: {violation_count}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        cv2.putText(frame, "[Press 't' to toggle Signal | 'q' to Quit]", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

        cv2.imshow("TRAFFIQ - Red Light Violation Enforcement", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            light_state = "RED" if light_state == "GREEN" else "GREEN"
            last_light_switch = time.time()
            print(f"[MANUAL OVERRIDE] Signal switched to {light_state}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_violation_detection(0)