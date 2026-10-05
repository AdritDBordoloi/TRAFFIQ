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
    """Sanitizes text by stripping spaces and keeping alphanumeric characters."""
    return re.sub(r'[^A-Za-z0-9]', '', raw_text).upper()

def get_vehicle_semi_crop(frame, box, scale=2.0, min_w=320, min_h=240):
    """
    Extracts a semi close-up photo of the vehicle with the vehicle centered in the frame.
    
    Args:
        frame: Full frame image as a numpy array.
        box: (x1, y1, x2, y2) bounding box coordinates of the vehicle.
        scale: Multiplier around vehicle bounding box for semi close-up framing.
        min_w: Minimum width of the crop to ensure high clarity for distant vehicles.
        min_h: Minimum height of the crop.
        
    Returns:
        Numpy array of the cropped image region.
    """
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    # Scale vehicle box dimensions for semi close-up framing (~50% subject occupancy)
    crop_w = int(w * scale)
    crop_h = int(h * scale)

    # Ensure clear photo resolution even if vehicle is detected at a distance
    crop_w = max(crop_w, min_w)
    crop_h = max(crop_h, min_h)

    # Maintain a balanced landscape aspect ratio (width >= 1.2 * height)
    if crop_w < int(crop_h * 1.2):
        crop_w = int(crop_h * 1.2)

    # Bound by overall frame size
    crop_w = min(width, crop_w)
    crop_h = min(height, crop_h)

    half_w = crop_w // 2
    half_h = crop_h // 2

    # Center the crop box on (cx, cy)
    c_x1 = cx - half_w
    c_x2 = c_x1 + crop_w
    c_y1 = cy - half_h
    c_y2 = c_y1 + crop_h

    # Slide window smoothly if near camera edges to keep full crop within bounds
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

def lookup_vehicle_owner(plate_number):
    """Matches detected plate against the RTO database."""
    conn = sqlite3.connect("traffiq_enforcement.db")
    cursor = conn.cursor()
    
    # 1. Direct exact match
    cursor.execute("""
        SELECT owner_name, contact_number, vehicle_model, registration_city 
        FROM vehicle_registry 
        WHERE plate_number = ?
    """, (plate_number,))
    row = cursor.fetchone()

    # 2. Substring fallback (handles slight OCR prefixes/suffixes)
    if not row and len(plate_number) >= 4:
        cursor.execute("""
            SELECT owner_name, contact_number, vehicle_model, registration_city 
            FROM vehicle_registry 
            WHERE plate_number LIKE ?
        """, (f"%{plate_number}%",))
        row = cursor.fetchone()

    conn.close()

    if row:
        return {
            "owner": row[0],
            "phone": row[1],
            "model": row[2],
            "city": row[3],
            "registered": True
        }
    return {
        "owner": "UNKNOWN / UNREGISTERED",
        "phone": "NOT ON FILE",
        "model": "Unverified",
        "city": "Unknown",
        "registered": False
    }

def init_enforcement_db():
    """Ensures database schema and columns for e-challan records exist."""
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
    cursor.execute("PRAGMA table_info(challan_records)")
    existing_cols = [row[1] for row in cursor.fetchall()]
    for col in ["contact_number", "vehicle_model", "full_evidence_path", "crop_evidence_path"]:
        if col not in existing_cols:
            cursor.execute(f"ALTER TABLE challan_records ADD COLUMN {col} TEXT DEFAULT ''")
    conn.commit()
    conn.close()

def issue_challan(track_id, plate, owner_data, full_img_path, crop_img_path, vehicle_dir=None):
    """Inserts challan into database and generates an electronic receipt text file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fine_amount = 1000  # Fine amount in INR

    # 1. Insert into SQLite Database
    conn = sqlite3.connect("traffiq_enforcement.db")
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO challan_records (
        timestamp, track_id, plate_number, owner_name, contact_number, 
        vehicle_model, violation_type, fine_amount, full_evidence_path, crop_evidence_path
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        timestamp, int(track_id), plate, owner_data["owner"], owner_data["phone"],
        owner_data["model"], "Red Light Signal Violation", fine_amount,
        full_img_path, crop_img_path
    ))
    challan_id = cursor.lastrowid
    conn.commit()
    conn.close()

    # 2. Save Printable Challan Notice File
    if vehicle_dir:
        receipt_filename = f"{vehicle_dir}/challan.txt"
    else:
        receipt_filename = f"violations/CHALLAN_{challan_id}_ID{track_id}.txt"
    with open(receipt_filename, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("       TRAFFIQ AUTOMATED TRAFFIC ENFORCEMENT NOTICE     \n")
        f.write("=" * 60 + "\n")
        f.write(f"Challan Number     : TRFQ-{challan_id:06d}\n")
        f.write(f"Violation DateTime : {timestamp}\n")
        f.write(f"Violation Type     : Red Light Disobedience (Section 119/177)\n")
        f.write(f"Vehicle Track ID   : #{track_id}\n")
        f.write(f"Detected Plate     : {plate}\n")
        f.write(f"Registered Owner   : {owner_data['owner']}\n")
        f.write(f"Contact Number     : {owner_data['phone']}\n")
        f.write(f"Vehicle Model      : {owner_data['model']} ({owner_data['city']})\n")
        f.write(f"Fine Amount Due    : INR {fine_amount}\n")
        f.write(f"Vehicle Evidence   : {full_img_path}\n")
        f.write(f"Plate Crop Evidence: {crop_img_path}\n")
        f.write("=" * 60 + "\n")

    print("\n" + "=" * 60)
    print(f" [!] CHALLAN ISSUED: TRFQ-{challan_id:06d} for ID:#{track_id}")
    print(f"     Plate: {plate} | Owner: {owner_data['owner']} ({owner_data['model']})")
    print(f"     Fine: INR {fine_amount} | Receipt: {receipt_filename}")
    print("=" * 60 + "\n")

    return challan_id

def run_enforcement_system(camera_index=0):
    init_enforcement_db()
    os.makedirs("violations", exist_ok=True)

    print("[INFO] Initializing YOLOv8s on RTX 4060 GPU...")
    yolo_model = YOLO("yolov8s.pt")

    print("[INFO] Initializing EasyOCR GPU Engine...")
    ocr_reader = easyocr.Reader(['en'], gpu=True)

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    track_positions = {}
    counted_ids = set()
    violated_ids = set()

    # Visual banner state
    recent_challan = None
    banner_display_time = 0

    # Traffic signal state (toggles every 8s or press 't')
    light_state = "GREEN"
    last_light_switch = time.time()
    light_cycle_duration = 8.0

    prev_time = time.time()
    print("[SUCCESS] TRAFFIQ Enforcement Active.")
    print("[CONTROLS] Press 't' to toggle Signal between RED / GREEN. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        stop_line_y = int(height * 0.55)

        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time

        # Automatic Signal Cycle
        if current_time - last_light_switch > light_cycle_duration:
            light_state = "RED" if light_state == "GREEN" else "GREEN"
            last_light_switch = current_time

        # Run ByteTrack Object Tracking
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

        # Draw Stop Line
        line_color = (0, 0, 255) if light_state == "RED" else (0, 255, 0)
        cv2.line(frame, (0, stop_line_y), (width, stop_line_y), line_color, 3)
        cv2.putText(frame, f"STOP LINE [{light_state}]", (10, stop_line_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, line_color, 2)

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

                # Detect when a vehicle crosses the stop line
                if prev_cy < stop_line_y <= cy:
                    if track_id not in counted_ids:
                        counted_ids.add(track_id)

                    # Trigger Violation if Signal is RED
                    if light_state == "RED" and track_id not in violated_ids:
                        violated_ids.add(track_id)
                        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

                        # Create dedicated folder for this violating vehicle
                        vehicle_dir = f"violations/vehicle_ID{track_id}_{timestamp_str}"
                        os.makedirs(vehicle_dir, exist_ok=True)

                        # 1. Capture Semi Close-Up Cropped Photo of Violating Vehicle (Centered)
                        vehicle_semi_crop = get_vehicle_semi_crop(frame, (x1, y1, x2, y2))
                        full_evidence_path = f"{vehicle_dir}/vehicle_image.jpg"
                        if vehicle_semi_crop.size > 0:
                            cv2.imwrite(full_evidence_path, vehicle_semi_crop)
                        else:
                            cv2.imwrite(full_evidence_path, frame)

                        # 2. Capture Zoomed Vehicle Crop
                        crop_y1 = max(0, y1 - 10)
                        crop_y2 = min(height, y2 + 10)
                        crop_x1 = max(0, x1 - 10)
                        crop_x2 = min(width, x2 + 10)
                        vehicle_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                        crop_evidence_path = f"{vehicle_dir}/plate_crop.jpg"
                        # 3. Read License Plate using EasyOCR
                        detected_plate = "UNKNOWN"
                        if vehicle_crop.size > 0:
                            cv2.imwrite(crop_evidence_path, vehicle_crop)
                            ocr_data = ocr_reader.readtext(vehicle_crop)
                            candidates = []
                            for (_, text, prob) in ocr_data:
                                cleaned = clean_plate_text(text)
                                if len(cleaned) >= 4 and prob > 0.3:
                                    candidates.append(cleaned)
                            
                            if candidates:
                                detected_plate = max(candidates, key=len)

                        # 4. Match with Database & Issue Challan inside vehicle folder
                        owner_info = lookup_vehicle_owner(detected_plate)
                        challan_id = issue_challan(
                            track_id, detected_plate, owner_info, 
                            full_evidence_path, crop_evidence_path,
                            vehicle_dir=vehicle_dir
                        )

                        # Store notification state for HUD popup
                        recent_challan = {
                            "id": challan_id,
                            "plate": detected_plate,
                            "owner": owner_info["owner"],
                            "model": owner_info["model"],
                            "fine": 1000
                        }
                        banner_display_time = time.time()

                # Bounding box colors
                if track_id in violated_ids:
                    box_color = (0, 0, 255)
                    box_label = f"VIOLATION #{track_id}"
                else:
                    box_color = (0, 255, 0) if track_id in counted_ids else (255, 150, 0)
                    box_label = f"ID:{track_id} {cls_name}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, box_label, (x1, max(20, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 2)

        # On-Screen Challan Notification Banner (Displays for 6 seconds after violation)
        if recent_challan and (time.time() - banner_display_time < 6.0):
            card_top = height - 110
            # Dark red background box with white border
            cv2.rectangle(frame, (20, card_top), (width - 20, height - 15), (0, 0, 160), -1)
            cv2.rectangle(frame, (20, card_top), (width - 20, height - 15), (255, 255, 255), 2)
            
            # Badge header
            cv2.putText(frame, f"CHALLAN ISSUED: TRFQ-{recent_challan['id']:06d} [FINE: INR {recent_challan['fine']}]", 
                        (40, card_top + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            
            # Owner & Vehicle Details
            details_str = f"Plate: {recent_challan['plate']}  |  Owner: {recent_challan['owner']}  |  Model: {recent_challan['model']}"
            cv2.putText(frame, details_str, (40, card_top + 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # Top Telemetry Box
        cv2.rectangle(frame, (10, 10), (380, 120), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (380, 120), line_color, 2)
        cv2.putText(frame, f"SIGNAL: {light_state}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, line_color, 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (230, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Flow Throughput: {len(counted_ids)}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Violations Logged: {len(violated_ids)}", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        cv2.imshow("TRAFFIQ - Automated Violation & E-Challan Enforcement", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            light_state = "RED" if light_state == "GREEN" else "GREEN"
            last_light_switch = time.time()
            print(f"[SIGNAL SWITCH] Manual toggle: {light_state}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_enforcement_system(0)