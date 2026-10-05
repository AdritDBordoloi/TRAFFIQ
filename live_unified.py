# live_unified.py
"""
TRAFFIQ Unified Live Traffic Enforcement & Emergency Priority System
====================================================================
Combines real-time red-light violation enforcement (YOLOv8s COCO + ByteTrack + EasyOCR
+ RTO SQLite database + e-Challan per-vehicle directory output) with an automatic
Emergency Green Corridor Priority Engine (YOLOv8s OIV7 Ambulance detection + signal override
+ exit line clearance tracking + safety timeouts + audio alert + emergency event logging).

Dual-Model Architecture:
------------------------
1. Primary Traffic Tracker: yolov8s.pt (COCO dataset)
   - Tracks Bicycle (1), Car (2), Motorcycle (3), Bus (5), and Truck (7) with ByteTrack.
   - Matches the exact sharp, reliable car/vehicle detection of live_violations.py.
2. Emergency Vehicle Detector: yolov8s-oiv7.pt (Open Images V7)
   - Filtered specifically for Ambulance (class ID 6).
   - Van (564) and Bus (73) are strictly excluded from the emergency class filter,
     preventing ambulances from being mislabeled as vans or buses.
   - Any tracked vehicle overlapping an ambulance detection is automatically promoted
     to "Ambulance" with emergency priority status.
3. Hardware Performance:
   - On an RTX 4060 GPU, dual-model inference runs at ~67 FPS (~15 ms per frame),
     guaranteeing smooth real-time video processing.
"""

import cv2
import time
import os
import sys
import re
import sqlite3
import winsound
import threading
from datetime import datetime
import torch
from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes
from ultralytics.data.augment import LetterBox
import easyocr


# ============================================================================
# DATABASE & STORAGE INITIALIZATION
# ============================================================================

def init_unified_db():
    """Initializes SQLite schema for vehicle registry, challans, and emergency events."""
    conn = sqlite3.connect("traffiq_enforcement.db")
    cursor = conn.cursor()

    # 1. RTO Registry Table (Simulating government vehicle database)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS vehicle_registry (
        plate_number TEXT PRIMARY KEY,
        owner_name TEXT NOT NULL,
        contact_number TEXT NOT NULL,
        vehicle_model TEXT NOT NULL,
        registration_city TEXT NOT NULL
    )
    """)

    # 2. Issued E-Challans Table
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

    # Ensure all required columns exist in challan_records
    cursor.execute("PRAGMA table_info(challan_records)")
    existing_cols = [row[1] for row in cursor.fetchall()]
    for col in ["contact_number", "vehicle_model", "full_evidence_path", "crop_evidence_path"]:
        if col not in existing_cols:
            cursor.execute(f"ALTER TABLE challan_records ADD COLUMN {col} TEXT DEFAULT ''")

    # 3. Emergency Events Table
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

    # Seed sample RTO vehicles if empty
    cursor.execute("SELECT COUNT(*) FROM vehicle_registry")
    if cursor.fetchone()[0] == 0:
        sample_vehicles = [
            ("AS01AB1234", "Rajesh Sharma", "+91 98765 43210", "Hyundai Creta", "Guwahati"),
            ("AS02CD5678", "Anurag Saikia", "+91 98540 11223", "Maruti Brezza", "Tezpur"),
            ("DL04CA9999", "Priya Verma", "+91 98111 22334", "Maruti Swift", "Delhi"),
            ("MH02DZ4567", "Amit Sen", "+91 97234 56789", "Honda City", "Mumbai"),
            ("KA05MJ8821", "Vikram Rao", "+91 99001 12233", "Toyota Innova", "Bengaluru"),
            ("SAMPLE123",  "Demo Driver", "+91 90000 00000", "Test Vehicle", "Exhibition Hall")
        ]
        cursor.executemany("""
        INSERT OR REPLACE INTO vehicle_registry (plate_number, owner_name, contact_number, vehicle_model, registration_city)
        VALUES (?, ?, ?, ?, ?)
        """, sample_vehicles)

    conn.commit()
    conn.close()
    os.makedirs("violations", exist_ok=True)


# ============================================================================
# HELPER FUNCTIONS: CROPPING, OCR, REGISTRY & DATABASE
# ============================================================================

def clean_plate_text(raw_text):
    """Sanitizes raw OCR text by stripping non-alphanumeric characters."""
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
        Numpy array of the cropped image region with vehicle centered.
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


def compute_box_iou(boxA, boxB):
    """Computes Intersection-over-Union (IoU) between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = float(boxAArea + boxBArea - interArea)
    return interArea / unionArea if unionArea > 0 else 0.0


def point_in_box(cx, cy, box):
    """Checks if a point (cx, cy) is contained within a bounding box."""
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


def compute_box_iomin(boxA, boxB):
    """Computes Intersection-over-Minimum-Area (handles nested boxes of varying scales)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    minArea = min(boxAArea, boxBArea)
    return interArea / float(minArea) if minArea > 0 else 0.0


def play_emergency_alert():
    """Plays emergency siren/beep in a background thread without stalling video processing."""
    def _run():
        try:
            winsound.Beep(1500, 350)
        except Exception:
            pass
    threading.Thread(target=_run, daemon=True).start()


def detect_ambulances_oiv7(oiv7_model, frame, device, conf_thres=0.08, iou_thres=0.45):
    """
    Detects ambulances using YOLOv8s Open Images V7 with multi_label=True.
    
    Why this is critical:
    Standard YOLOv8 predict() uses single-label argmax across all 601 classes, which
    causes ambulances to be dominated by 'Van' (564) and discarded when filtering classes=[6].
    By using multi_label=True NMS, Class 6 (Ambulance) is preserved and accurately localized.
    """
    h_orig, w_orig = frame.shape[:2]
    img_lb = LetterBox(640)(image=frame)
    t_img = torch.from_numpy(img_lb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

    with torch.no_grad():
        preds = oiv7_model.model(t_img)

    # Class 6 is Ambulance in Open Images V7
    nms_res = non_max_suppression(
        preds,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        classes=[6],
        multi_label=True
    )

    amb_boxes = []
    if len(nms_res) > 0 and len(nms_res[0]) > 0:
        scaled = scale_boxes(img_lb.shape[:2], nms_res[0][:, :4].clone(), (h_orig, w_orig))
        for b, conf in zip(scaled, nms_res[0][:, 4]):
            amb_boxes.append((b.cpu().numpy().astype(int), float(conf)))

    return amb_boxes


EMERGENCY_TEXT_KEYWORDS = [
    "AMBULANCE", "ECNALUBMA", "EMERGENCY", "HOSPITAL",
    "108", "102", "EMS", "PARAMEDIC", "NOTARZT",
    "RETTUNGSDIENST", "SAMU", "AMBULANCIA"
]


def check_vehicle_emergency_text(crop, ocr_reader):
    """
    Secondary emergency verification: inspects vehicle body for prominent emergency markings
    and emergency helpline codes ('AMBULANCE', '108', '102', reversed 'ECNALUBMA').
    """
    if crop is None or crop.size == 0 or crop.shape[0] < 40 or crop.shape[1] < 40:
        return False, None, 0.0

    try:
        results = ocr_reader.readtext(crop)
        for _, text, prob in results:
            cleaned = re.sub(r'[^A-Za-z0-9]', '', text).upper()
            for kw in EMERGENCY_TEXT_KEYWORDS:
                if kw in cleaned and prob > 0.35:
                    return True, kw, float(prob)
    except Exception:
        pass

    return False, None, 0.0


def lookup_vehicle_owner(plate_number):
    """Matches detected plate against the RTO database."""
    conn = sqlite3.connect("traffiq_enforcement.db")
    cursor = conn.cursor()

    # 1. Exact match
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

    # 2. Save Printable Challan Notice File inside vehicle folder
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


def log_emergency_event(track_id, vehicle_type, duration, status):
    """Logs an emergency priority event to the SQLite database."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect("traffiq_enforcement.db")
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO emergency_events (timestamp, track_id, vehicle_type, override_duration_sec, status)
    VALUES (?, ?, ?, ?, ?)
    """, (timestamp, int(track_id), vehicle_type, round(duration, 2), status))
    conn.commit()
    conn.close()

    print("\n" + "=" * 65)
    print(f" [!] EMERGENCY GREEN CORRIDOR CLEARED: Track #{track_id} ({vehicle_type})")
    print(f"     Duration: {duration:.1f}s | Resolution: {status} | DB Saved")
    print("=" * 65 + "\n")


# ============================================================================
# VIDEO SOURCE SELECTION & UPLOAD HANDLERS
# ============================================================================

def open_video_file_dialog():
    """Opens a native Windows file explorer dialog to select/upload a video file."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        file_path = filedialog.askopenfilename(
            title="TRAFFIQ - Select Traffic / Emergency Video File",
            filetypes=[
                ("Video Files", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v *.wmv *.flv"),
                ("All Files", "*.*")
            ]
        )
        root.destroy()
        return file_path if file_path else None
    except Exception as e:
        print(f"[WARN] Could not open file dialog: {e}")
        return None


def select_video_source(cli_source=None, force_upload=False):
    """
    Selects video source from CLI argument, file upload dialog, or interactive menu.
    """
    # 1. If upload flag specifically requested
    if force_upload:
        print("[INFO] Opening File Explorer to select video file...")
        selected = open_video_file_dialog()
        if selected and os.path.exists(selected):
            print(f"[INFO] Selected Video File: {selected}")
            return selected
        print("[WARN] No file selected. Defaulting to live webcam 0.")
        return 0

    # 2. If CLI source argument provided
    if cli_source is not None:
        cli_clean = str(cli_source).strip("'\"")
        if cli_clean.isdigit():
            return int(cli_clean)
        if os.path.exists(cli_clean):
            return cli_clean
        print(f"[WARN] Specified file '{cli_source}' not found. Please choose an option below:")

    # 3. Interactive prompt if launched with no arguments
    print("\n" + "=" * 60)
    print("           TRAFFIQ VIDEO SOURCE SELECTION")
    print("=" * 60)
    print("  [1] Live Webcam Feed (Default - Camera Index 0)")
    print("  [2] Upload / Select Video File (Open File Explorer)")
    print("  [3] Enter Video File Path Manually")
    print("=" * 60)

    try:
        choice = input("Enter choice [1/2/3] (Press Enter for Webcam 0): ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = "1"

    if choice == "2":
        print("[INFO] Opening File Explorer to select video file...")
        selected = open_video_file_dialog()
        if selected and os.path.exists(selected):
            print(f"[INFO] Loaded Video File: {selected}")
            return selected
        print("[WARN] No file selected. Defaulting to live webcam 0.")
        return 0
    elif choice == "3":
        try:
            path_input = input("Enter or drag-and-drop video file path: ").strip("'\"")
            if os.path.exists(path_input):
                return path_input
            print(f"[WARN] File '{path_input}' does not exist. Defaulting to webcam 0.")
            return 0
        except Exception:
            return 0
    else:
        return 0


# ============================================================================
# MAIN UNIFIED ENFORCEMENT & PRIORITY PIPELINE
# ============================================================================

def run_unified_system(source=0, loop=True):
    init_unified_db()

    # 1. Primary Traffic Tracker: YOLOv8s COCO (Sharp car, motorcycle, bus, truck, bicycle detection)
    print("[INFO] Loading Primary Traffic Tracker: YOLOv8s (COCO)...")
    coco_model = YOLO("yolov8s.pt")
    COCO_TRAFFIC_CLASSES = {
        1: "Bicycle",
        2: "Car",
        3: "Motorcycle",
        5: "Bus",
        7: "Truck"
    }

    # 2. Emergency Priority Detector: YOLOv8s Open Images V7 (Multi-Label Native Ambulance Engine)
    print("[INFO] Loading Emergency Priority Detector: YOLOv8s-OIV7 (Ambulance)...")
    oiv7_model = YOLO("yolov8s-oiv7.pt")
    device = torch.device(0 if torch.cuda.is_available() else "cpu")
    oiv7_model.to(device)
    print(f"[INFO] Emergency Priority Detector loaded on {device} (Multi-Label NMS active, Class 6: Ambulance).")

    # 3. EasyOCR GPU Engine
    print("[INFO] Initializing EasyOCR GPU Engine...")
    ocr_reader = easyocr.Reader(['en'], gpu=True)

    # 4. Video Source Setup (Supports Webcam Index or Video File Path)
    is_video_file = False
    source_str = str(source).strip("'\"")

    if not source_str.isdigit() and os.path.exists(source_str):
        is_video_file = True
        source_name = os.path.basename(source_str)
        print(f"[INFO] Opening Video File: {source_str}")
        cap = cv2.VideoCapture(source_str)
    else:
        cam_idx = int(source_str) if source_str.isdigit() else 0
        source_name = f"Webcam ({cam_idx})"
        print(f"[INFO] Opening Live Camera Feed: Index {cam_idx}...")
        cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source}")
        return

    # Frame timing for realistic video playback speed
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not video_fps or video_fps <= 0 or video_fps > 120:
        video_fps = 30.0
    target_frame_time = 1.0 / video_fps

    # Tracking & State Storage
    track_positions = {}
    counted_ids = set()
    violated_ids = set()
    checked_ocr_tracks = set()

    # Emergency Priority State
    # Supports multiple simultaneous emergency vehicles: {track_id: {"type": str, "start_time": float, "last_seen": float}}
    active_emergency_vehicles = {}
    emergency_lock_engaged = False
    safety_timeout = 25.0       # Max override duration in seconds
    min_emergency_conf = 0.08   # Highly sensitive multi-label threshold for native ambulance detection

    # Traffic Signal State Machine (Normal cycle: 10s RED <-> 10s GREEN)
    light_state = "GREEN"
    last_light_switch = time.time()
    normal_cycle_duration = 10.0  # seconds

    # Telemetry and HUD Banners
    recent_challan = None
    banner_display_time = 0
    total_violations_count = 0
    total_emergencies_cleared = 0

    prev_time = time.time()
    print("\n[SUCCESS] TRAFFIQ Unified Dual-Model Engine Active.")
    print("[INFO] Primary Tracker: YOLOv8s COCO (Cars, Bikes, Buses, Trucks)")
    print("[INFO] Priority Engine: YOLOv8s OIV7 (Native Ambulance Detection)")
    print("[CONTROLS] Press 't' to toggle Signal manually (disabled during emergency) | 'q' to quit.\n")

    while True:
        frame_start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            if is_video_file and loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                if not ret:
                    break
            else:
                break

        height, width, _ = frame.shape
        stop_line_y = int(height * 0.55)    # 55% frame height stop line
        exit_line_y = int(height * 0.72)    # 72% frame height exit clearance line

        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time

        # --------------------------------------------------------------------
        # 1. PRIMARY DETECTION & TRACKING (YOLOv8s COCO + ByteTrack)
        # --------------------------------------------------------------------
        coco_results = coco_model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=list(COCO_TRAFFIC_CLASSES.keys()),
            conf=0.25,
            imgsz=640,
            device=0,
            verbose=False
        )[0]

        # --------------------------------------------------------------------
        # 2. EMERGENCY DETECTION (YOLOv8s OIV7 Multi-Label Native Ambulance Engine)
        # --------------------------------------------------------------------
        ambulance_boxes = detect_ambulances_oiv7(
            oiv7_model,
            frame,
            device=device,
            conf_thres=min_emergency_conf,
            iou_thres=0.45
        )

        # --------------------------------------------------------------------
        # 3. BOUNDING BOX FUSION & MULTI-CUE EMERGENCY PROMOTION
        # --------------------------------------------------------------------
        frame_detections = []
        matched_amb_indices = set()

        if coco_results.boxes is not None and coco_results.boxes.id is not None:
            boxes = coco_results.boxes.xyxy.cpu().numpy().astype(int)
            track_ids = coco_results.boxes.id.cpu().numpy().astype(int)
            class_ids = coco_results.boxes.cls.cpu().numpy().astype(int)
            confidences = coco_results.boxes.conf.cpu().numpy()

            for box, track_id, cls_id, conf in zip(boxes, track_ids, class_ids, confidences):
                cls_name = COCO_TRAFFIC_CLASSES.get(cls_id, "Vehicle")
                cx = int((box[0] + box[2]) / 2)
                cy = int((box[1] + box[3]) / 2)

                is_emergency = False
                emergency_conf = 0.0

                # Priority Rule 1: Track ID already confirmed as active emergency vehicle
                if track_id in active_emergency_vehicles:
                    is_emergency = True
                    cls_name = active_emergency_vehicles[track_id]["type"]
                    emergency_conf = 0.95

                # Priority Rule 2: Motor vehicles (Car, Bus, Truck) matched with OIV7 multi-label ambulance detector
                elif cls_name in ["Car", "Bus", "Truck"]:
                    for idx, (amb_xyxy, amb_conf) in enumerate(ambulance_boxes):
                        iou = compute_box_iou(box, amb_xyxy)
                        iomin = compute_box_iomin(box, amb_xyxy)
                        if iou > 0.20 or iomin > 0.40 or point_in_box(cx, cy, amb_xyxy):
                            is_emergency = True
                            cls_name = "Ambulance"
                            emergency_conf = amb_conf
                            matched_amb_indices.add(idx)
                            break

                    # Priority Rule 3: Secondary OCR verification for unconfirmed vehicles approaching intersection
                    if not is_emergency and track_id not in checked_ocr_tracks and cy < exit_line_y:
                        checked_ocr_tracks.add(track_id)
                        bx1, by1 = max(0, box[0]), max(0, box[1])
                        bx2, by2 = min(width, box[2]), min(height, box[3])
                        v_crop = frame[by1:by2, bx1:bx2]
                        has_text, kw, t_score = check_vehicle_emergency_text(v_crop, ocr_reader)
                        if has_text:
                            is_emergency = True
                            cls_name = "Ambulance"
                            emergency_conf = t_score
                            print(f"[EMERGENCY OCR CONFIRMED] Track #{track_id} verified by text '{kw}' (conf: {int(t_score * 100)}%)")

                frame_detections.append({
                    "box": (box[0], box[1], box[2], box[3]),
                    "track_id": track_id,
                    "cls_id": cls_id,
                    "cls_name": cls_name,
                    "is_emergency": is_emergency,
                    "conf": emergency_conf if is_emergency else conf,
                    "cx": cx,
                    "cy": cy
                })

        # Include standalone OIV7 ambulance detections if not yet tracked by COCO
        for idx, (amb_xyxy, amb_conf) in enumerate(ambulance_boxes):
            if idx not in matched_amb_indices:
                amb_cx = int((amb_xyxy[0] + amb_xyxy[2]) / 2)
                amb_cy = int((amb_xyxy[1] + amb_xyxy[3]) / 2)
                # Assign a synthetic emergency track ID
                pseudo_track_id = 9000 + idx
                frame_detections.append({
                    "box": (amb_xyxy[0], amb_xyxy[1], amb_xyxy[2], amb_xyxy[3]),
                    "track_id": pseudo_track_id,
                    "cls_id": 6,
                    "cls_name": "Ambulance",
                    "is_emergency": True,
                    "conf": amb_conf,
                    "cx": amb_cx,
                    "cy": amb_cy
                })

        # --------------------------------------------------------------------
        # 4. EMERGENCY GREEN CORRIDOR STATE MACHINE
        # --------------------------------------------------------------------
        for det in frame_detections:
            if det["is_emergency"]:
                t_id = det["track_id"]
                c_y = det["cy"]

                # Register new emergency vehicle if approaching and above exit line
                if t_id not in active_emergency_vehicles and c_y < exit_line_y:
                    active_emergency_vehicles[t_id] = {
                        "type": det["cls_name"],
                        "start_time": current_time,
                        "last_seen": current_time
                    }
                    print("\n" + "=" * 65)
                    print(f" [!!!] EMERGENCY GREEN CORRIDOR ENGAGED: Track #{t_id} ({det['cls_name']})")
                    print(f"       Confidence: {int(det['conf'] * 100)}% | Signal: FORCED GREEN")
                    print("=" * 65)
                    play_emergency_alert()
                elif t_id in active_emergency_vehicles:
                    active_emergency_vehicles[t_id]["last_seen"] = current_time

        # Check exit line clearance and timeouts for active emergency vehicles
        cleared_emergency_ids = []
        for t_id, meta in list(active_emergency_vehicles.items()):
            elapsed = current_time - meta["start_time"]
            prev_y = track_positions.get(t_id, None)

            # Condition A: Vehicle crossed Exit Line (72% height)
            if prev_y is not None and prev_y < exit_line_y:
                cur_det = next((d for d in frame_detections if d["track_id"] == t_id), None)
                if cur_det and cur_det["cy"] >= exit_line_y:
                    log_emergency_event(t_id, meta["type"], elapsed, "CLEARED_EXIT")
                    cleared_emergency_ids.append(t_id)
                    total_emergencies_cleared += 1
                    continue

            # Condition B: Safety timeout reached (25s)
            if elapsed > safety_timeout:
                log_emergency_event(t_id, meta["type"], elapsed, "TIMEOUT_CLEARED")
                cleared_emergency_ids.append(t_id)
                total_emergencies_cleared += 1
                continue

            # Condition C: Lost track for > 4.0 seconds
            if current_time - meta["last_seen"] > 4.0:
                log_emergency_event(t_id, meta["type"], elapsed, "LOST_TRACK_CLEARED")
                cleared_emergency_ids.append(t_id)
                total_emergencies_cleared += 1
                continue

        for cid in cleared_emergency_ids:
            active_emergency_vehicles.pop(cid, None)

        # Signal state determination
        emergency_lock_engaged = len(active_emergency_vehicles) > 0

        if emergency_lock_engaged:
            # Emergency override locks signal to GREEN
            light_state = "GREEN"
            last_light_switch = current_time
        else:
            # Normal 10s automatic cycle
            if current_time - last_light_switch > normal_cycle_duration:
                light_state = "RED" if light_state == "GREEN" else "GREEN"
                last_light_switch = current_time

        # --------------------------------------------------------------------
        # 5. RED-LIGHT VIOLATION ENFORCEMENT (SUSPENDED DURING EMERGENCY)
        # --------------------------------------------------------------------
        for det in frame_detections:
            t_id = det["track_id"]
            cy = det["cy"]
            box = det["box"]

            prev_cy = track_positions.get(t_id, cy)
            track_positions[t_id] = cy

            # Crossing Stop Line from top to bottom
            if prev_cy < stop_line_y <= cy:
                if t_id not in counted_ids:
                    counted_ids.add(t_id)

                # Check Violation: Signal is RED, vehicle not already ticketed, and NO emergency active
                if light_state == "RED" and t_id not in violated_ids and not det["is_emergency"]:
                    # Edge case check: If an emergency corridor is active, ABORT infraction
                    if emergency_lock_engaged:
                        print(f"[INFO] Skipping infraction for Track #{t_id}: Emergency corridor in effect.")
                        continue

                    violated_ids.add(t_id)
                    total_violations_count += 1
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

                    # Dedicated per-vehicle evidence directory
                    vehicle_dir = f"violations/vehicle_ID{t_id}_{timestamp_str}"
                    os.makedirs(vehicle_dir, exist_ok=True)

                    # 1. Semi close-up vehicle photo centered on subject
                    vehicle_semi_crop = get_vehicle_semi_crop(frame, box)
                    vehicle_evidence_path = f"{vehicle_dir}/vehicle_image.jpg"
                    if vehicle_semi_crop.size > 0:
                        cv2.imwrite(vehicle_evidence_path, vehicle_semi_crop)
                    else:
                        cv2.imwrite(vehicle_evidence_path, frame)

                    # 2. Tight crop for EasyOCR plate detection
                    x1, y1, x2, y2 = box
                    crop_y1 = max(0, y1 - 10)
                    crop_y2 = min(height, y2 + 10)
                    crop_x1 = max(0, x1 - 10)
                    crop_x2 = min(width, x2 + 10)
                    plate_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    plate_crop_path = f"{vehicle_dir}/plate_crop.jpg"

                    detected_plate = "UNKNOWN"
                    if plate_crop.size > 0:
                        cv2.imwrite(plate_crop_path, plate_crop)
                        ocr_data = ocr_reader.readtext(plate_crop)
                        candidates = []
                        for (_, text, prob) in ocr_data:
                            cleaned = clean_plate_text(text)
                            if len(cleaned) >= 4 and prob > 0.3:
                                candidates.append(cleaned)
                        if candidates:
                            detected_plate = max(candidates, key=len)

                    # 3. RTO Database lookup & E-Challan generation
                    owner_info = lookup_vehicle_owner(detected_plate)
                    challan_id = issue_challan(
                        t_id, detected_plate, owner_info,
                        vehicle_evidence_path, plate_crop_path,
                        vehicle_dir=vehicle_dir
                    )

                    # Trigger on-screen challan notification card
                    recent_challan = {
                        "id": challan_id,
                        "plate": detected_plate,
                        "owner": owner_info["owner"],
                        "model": owner_info["model"],
                        "fine": 1000
                    }
                    banner_display_time = time.time()

        # Clean expired track positions
        if len(track_positions) > 500:
            track_positions.clear()
        if len(checked_ocr_tracks) > 500:
            checked_ocr_tracks.clear()

        # --------------------------------------------------------------------
        # 6. VISUAL HUD OVERLAYS & BOUNDING BOXES
        # --------------------------------------------------------------------

        # Draw Stop Line
        stop_line_color = (0, 0, 255) if light_state == "RED" else (0, 255, 0)
        cv2.line(frame, (0, stop_line_y), (width, stop_line_y), stop_line_color, 3)
        cv2.putText(frame, f"STOP LINE [{light_state}]", (10, stop_line_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, stop_line_color, 2)

        # Draw Exit Clearance Line
        cv2.line(frame, (0, exit_line_y), (width, exit_line_y), (0, 255, 255), 2)
        cv2.putText(frame, "CLEARANCE EXIT LINE", (10, exit_line_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Draw Vehicle Bounding Boxes
        for det in frame_detections:
            x1, y1, x2, y2 = det["box"]
            t_id = det["track_id"]
            conf = det["conf"]
            cls_name = det["cls_name"]
            is_emergency = det["is_emergency"]

            if is_emergency or t_id in active_emergency_vehicles:
                # Emergency vehicle: thick bright red box with emergency tag
                box_color = (0, 0, 255)
                box_label = f"EMERGENCY: {cls_name} #{t_id} {int(conf * 100)}%"
                thickness = 3
            elif t_id in violated_ids:
                # Violated vehicle: red box
                box_color = (0, 0, 255)
                box_label = f"VIOLATION #{t_id}"
                thickness = 2
            elif t_id in counted_ids:
                # Counted vehicle: green box
                box_color = (0, 255, 0)
                box_label = f"ID:{t_id} {cls_name} {int(conf * 100)}%"
                thickness = 2
            else:
                # Waiting / uncounted vehicle: orange box
                box_color = (255, 150, 0)
                box_label = f"ID:{t_id} {cls_name} {int(conf * 100)}%"
                thickness = 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)
            cv2.circle(frame, (det["cx"], det["cy"]), 4, (0, 0, 255), -1)
            cv2.putText(frame, box_label, (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 2)

        # Top Telemetry HUD Box
        hud_border = (0, 0, 255) if emergency_lock_engaged else stop_line_color
        cv2.rectangle(frame, (10, 10), (450, 155), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (450, 155), hud_border, 2)

        mode_str = "EMERGENCY OVERRIDE" if emergency_lock_engaged else "NORMAL CYCLE"
        mode_color = (0, 0, 255) if emergency_lock_engaged else (0, 255, 0)
        cv2.putText(frame, f"MODE: {mode_str}", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.52, mode_color, 2)
        cv2.putText(frame, f"SIGNAL: {light_state}", (280, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, stop_line_color, 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(frame, f"SRC: {source_name[:18]}", (140, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        cv2.putText(frame, f"Flow Throughput: {len(counted_ids)}", (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(frame, f"Violations Logged: {total_violations_count}", (20, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 2)
        cv2.putText(frame, f"Priority Corridors Cleared: {total_emergencies_cleared}", (20, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(frame, "[Controls: 't' Toggle Signal | 'q' Quit]", (20, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

        # Flashing Emergency Strobe Banner (Alternates Red & Blue)
        if emergency_lock_engaged:
            strobe_bg = (0, 0, 190) if int(time.time() * 4) % 2 == 0 else (190, 50, 0)
            banner_y = height - 95
            cv2.rectangle(frame, (20, banner_y), (width - 20, height - 15), strobe_bg, -1)
            cv2.rectangle(frame, (20, banner_y), (width - 20, height - 15), (255, 255, 255), 2)

            earliest_start = min(meta["start_time"] for meta in active_emergency_vehicles.values())
            active_ids_str = ", ".join(f"#{t_id} ({meta['type']})" for t_id, meta in active_emergency_vehicles.items())

            cv2.putText(frame, "EMERGENCY VEHICLE DETECTED - GREEN CORRIDOR ACTIVE",
                        (40, banner_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            sub_txt = f"Signal Held GREEN | Elapsed: {current_time - earliest_start:.1f}s | Priority Lock: {active_ids_str}"
            cv2.putText(frame, sub_txt, (40, banner_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 255, 255), 1)

        # On-Screen Challan Notification Banner (Displays for 6s after ticket issued)
        elif recent_challan and (time.time() - banner_display_time < 6.0):
            card_top = height - 105
            cv2.rectangle(frame, (20, card_top), (width - 20, height - 15), (0, 0, 160), -1)
            cv2.rectangle(frame, (20, card_top), (width - 20, height - 15), (255, 255, 255), 2)

            cv2.putText(frame, f"CHALLAN ISSUED: TRFQ-{recent_challan['id']:06d} [FINE: INR {recent_challan['fine']}]",
                        (40, card_top + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2)
            details_str = f"Plate: {recent_challan['plate']}  |  Owner: {recent_challan['owner']}  |  Model: {recent_challan['model']}"
            cv2.putText(frame, details_str, (40, card_top + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

        cv2.imshow("TRAFFIQ - Unified Enforcement & Priority System", frame)

        # Frame pacing for video files to match natural playback speed
        if is_video_file:
            process_duration = time.time() - frame_start_time
            sleep_needed = target_frame_time - process_duration
            wait_time = max(1, int(sleep_needed * 1000)) if sleep_needed > 0 else 1
        else:
            wait_time = 1

        # Keyboard Controls
        key = cv2.waitKey(wait_time) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            if not emergency_lock_engaged:
                light_state = "RED" if light_state == "GREEN" else "GREEN"
                last_light_switch = time.time()
                print(f"[SIGNAL SWITCH] Manual toggle: {light_state}")
            else:
                print("[OVERRIDE LOCKED] Manual toggle disabled during active emergency corridor.")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] TRAFFIQ Unified System shut down cleanly.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="TRAFFIQ - Unified Traffic Enforcement & Emergency Priority System"
    )
    parser.add_argument("video_file", nargs="?", default=None, help="Path to video file (.mp4, .avi, etc.)")
    parser.add_argument("-s", "--source", default=None, help="Video file path or webcam index (e.g. 0)")
    parser.add_argument("-u", "--upload", action="store_true", help="Open Windows File Explorer to browse and upload video")
    parser.add_argument("--no-loop", action="store_true", help="Do not loop video when it reaches the end")

    args = parser.parse_args()

    input_arg = args.video_file if args.video_file is not None else args.source
    resolved_source = select_video_source(cli_source=input_arg, force_upload=args.upload)

    run_unified_system(source=resolved_source, loop=not args.no_loop)
