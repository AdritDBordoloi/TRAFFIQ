# live_enforcement.py
import cv2
import time
import os
import sys
import re
import sqlite3
from datetime import datetime
from ultralytics import YOLO
import easyocr

TRAFFIC_CLASSES = {
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

def clean_plate_text(raw_text):
    """Filters string to uppercase alphanumeric characters only."""
    cleaned = re.sub(r'[^A-Za-z0-9]', '', raw_text).upper()
    return cleaned

def lookup_owner(plate_number):
    """Queries SQLite registry for vehicle owner records."""
    conn = sqlite3.connect("traffiq_enforcement.db")
    cursor = conn.cursor()
    cursor.execute("SELECT owner_name, contact_number, vehicle_model FROM vehicle_registry WHERE plate_number = ?", (plate_number,))
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return {"owner": result[0], "phone": result[1], "model": result[2], "registered": True}
    return {"owner": "UNKNOWN / OUT-OF-STATE", "phone": "N/A", "model": "Unregistered", "registered": False}

def record_challan(track_id, plate, owner_info, evidence_path):
    """Logs the violation into the database and prints ticket details."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fine_amount = 1000  # Standard red light fine in INR

    conn = sqlite3.connect("traffiq_enforcement.db")
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO challan_records (timestamp, track_id, plate_number, owner_name, violation_type, fine_amount, evidence_path)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (timestamp, track_id, plate, owner_info["owner"], "Red Light Jump", fine_amount, evidence_path))
    conn.commit()
    conn.close()

    print("\n" + "=" * 55)
    print("           AUTOMATED E-CHALLAN ISSUED            ")
    print("=" * 55)
    print(f"Timestamp      : {timestamp}")
    print(f"Tracking ID    : #{track_id}")
    print(f"License Plate  : {plate}")
    print(f"Registered To  : {owner_info['owner']}")
    print(f"Contact No.    : {owner_info['phone']}")
    print(f"Vehicle Model  : {owner_info['model']}")
    print(f"Violation      : Red Light Signal Disobedience")
    print(f"Penalty Fine   : INR {fine_amount}")
    print(f"Evidence File  : {evidence_path}")
    print("=" * 55 + "\n")

def run_enforcement_system(camera_index=0):
    os.makedirs("violations", exist_ok=True)

    print("[INFO] Loading YOLOv8s on RTX 4060...")
    yolo_model = YOLO("yolov8s.pt")

    print("[INFO] Initializing EasyOCR GPU engine...")
    ocr_reader = easyocr.Reader(['en'], gpu=True)

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    track_positions = {}
    counted_ids = set()
    violated_ids = set()

    # On-screen visual alert memory
    recent_challan_alert = None
    alert_display_timer = 0

    # Traffic signal state management
    light_state = "GREEN"
    last_light_switch = time.time()
    light_cycle_duration = 8.0

    prev_time = time.time()
    print("[SUCCESS] Enforcement engine active. Press 't' to toggle light, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        stop_line_y = int(height * 0.55)

        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time

        # Cycle traffic signal
        if current_time - last_light_switch > light_cycle_duration:
            light_state = "RED" if light_state == "GREEN" else "GREEN"
            last_light_switch = current_time

        # Run ByteTrack object tracking
        results = yolo_model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=list(TRAFFIC_CLASSES.keys()),
            conf=0.25,
            imgsz=640,
            device=0,
            verbose=False
        )[0]

        # Render stop line
        line_color = (0, 0, 255) if light_state == "RED" else (0, 255, 0)
        cv2.line(frame, (0, stop_line_y), (width, stop_line_y), line_color, 3)
        cv2.putText(frame, f"STOP LINE [{light_state}]", (10, stop_line_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, line_color, 2)

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            class_ids = results.boxes.cls.cpu().numpy().astype(int)

            for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
                x1, y1, x2, y2 = box
                cls_name = TRAFFIC_CLASSES.get(cls_id, "Vehicle")
                cy = int((y1 + y2) / 2)
                cx = int((x1 + x2) / 2)

                if track_id not in track_positions:
                    track_positions[track_id] = cy
                prev_cy = track_positions[track_id]
                track_positions[track_id] = cy

                # Check line crossing
                if prev_cy < stop_line_y <= cy:
                    if track_id not in counted_ids:
                        counted_ids.add(track_id)

                    # Trigger violation enforcement on RED
                    if light_state == "RED" and track_id not in violated_ids:
                        violated_ids.add(track_id)

                        # 1. Crop vehicle evidence
                        crop_y1 = max(0, y1 - 10)
                        crop_y2 = min(height, y2 + 10)
                        crop_x1 = max(0, x1 - 10)
                        crop_x2 = min(width, x2 + 10)
                        vehicle_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                        # 2. Save evidence image
                        timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
                        evidence_path = f"violations/violation_ID{track_id}_{timestamp_file}.jpg"
                        cv2.imwrite(evidence_path, vehicle_crop)

                        # 3. OCR character recognition on the crop
                        detected_plate = "UNKNOWN"
                        ocr_results = ocr_reader.readtext(vehicle_crop)
                        
                        # Find best alphanumeric candidate
                        candidates = []
                        for (_, text, prob) in ocr_results:
                            cleaned = clean_plate_text(text)
                            if len(cleaned) >= 4 and prob > 0.3:
                                candidates.append(cleaned)
                        
                        if candidates:
                            # Pick longest alphanumeric match
                            detected_plate = max(candidates, key=len)

                        # 4. Query RTO Database for Owner
                        owner_details = lookup_owner(detected_plate)

                        # 5. Issue Challan Record
                        record_challan(track_id, detected_plate, owner_details, evidence_path)

                        # Trigger on-screen banner
                        recent_challan_alert = {
                            "plate": detected_plate,
                            "owner": owner_details["owner"],
                            "fine": 1000
                        }
                        alert_display_timer = time.time()

                # Visual bounding box
                box_color = (0, 0, 255) if track_id in violated_ids else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                
                status_txt = f"VIOLATION #{track_id}" if track_id in violated_ids else f"ID:{track_id} {cls_name}"
                cv2.putText(frame, status_txt, (x1, max(20, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 2)

        # Draw real-time Challan Pop-Up Banner when a violation occurs
        if recent_challan_alert and (time.time() - alert_display_timer < 5.0):
            banner_y = height - 100
            cv2.rectangle(frame, (20, banner_y), (width - 20, height - 20), (0, 0, 180), -1)
            cv2.rectangle(frame, (20, banner_y), (width - 20, height - 20), (255, 255, 255), 2)
            cv2.putText(frame, "AUTOMATED E-CHALLAN GENERATED", (40, banner_y + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, f"Plate: {recent_challan_alert['plate']}  |  Owner: {recent_challan_alert['owner']}  |  Fine: INR {recent_challan_alert['fine']}",
                        (40, banner_y + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 255), 1)

        # Upper Telemetry HUD
        cv2.rectangle(frame, (10, 10), (370, 115), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (370, 115), line_color, 2)
        cv2.putText(frame, f"SIGNAL: {light_state}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, line_color, 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (220, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Total Passed: {len(counted_ids)}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"E-Challans Issued: {len(violated_ids)}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        cv2.imshow("TRAFFIQ - Automated Enforcement & E-Challan System", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            light_state = "RED" if light_state == "GREEN" else "GREEN"
            last_light_switch = time.time()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_enforcement_system(0)