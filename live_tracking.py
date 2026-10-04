import cv2
import time
import sys
from ultralytics import YOLO

# Road traffic classes from COCO dataset
TRAFFIC_CLASSES = {
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

def run_traffic_tracking(camera_index=0):
    print("[INFO] Initializing YOLOv8 Small (yolov8s.pt) with ByteTrack...")
    # Automatically downloads yolov8s.pt (~22 MB) on first launch
    model = YOLO("yolov8s.pt")

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
    
    # Optional: Request 1280x720 from webcam for cleaner image details
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {camera_index}.")
        return

    # Track historical center points: {track_id: cy}
    track_positions = {}
    
    # Store IDs that have already crossed the tripwire to prevent duplicate counts
    counted_ids = set()
    
    # Cumulative vehicle statistics
    cumulative_counts = {"Car": 0, "Motorcycle": 0, "Bus": 0, "Truck": 0, "Bicycle": 0}

    prev_time = time.time()
    print("[SUCCESS] Tracking pipeline active on GPU. Press 'q' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera feed lost.")
            break

        height, width, _ = frame.shape
        line_y = int(height * 0.55)  # Counting tripwire at 55% frame height

        # Compute FPS
        now = time.time()
        fps = 1 / (now - prev_time) if (now - prev_time) > 0 else 0
        prev_time = now

        # Run ByteTrack with YOLOv8 Small, explicit imgsz=640, conf=0.25 on RTX 4060
        results = model.track(
            source=frame,
            persist=True,               # Preserves track IDs across frames
            tracker="bytetrack.yaml",   # ByteTrack tracker configuration
            classes=list(TRAFFIC_CLASSES.keys()),
            conf=0.25,                  # Lowers detection dropout
            imgsz=640,                  # Fixed 640x640 inference resolution
            device=0,                   # Targets RTX 4060 GPU
            verbose=False
        )[0]

        current_active_density = 0

        # Draw virtual counting line (Yellow)
        cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 255), 2)
        cv2.putText(frame, "COUNTING TRIPWIRE", (10, line_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Process detections with assigned tracking IDs
        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            class_ids = results.boxes.cls.cpu().numpy().astype(int)
            confidences = results.boxes.conf.cpu().numpy()

            current_active_density = len(track_ids)

            for box, track_id, cls_id, conf in zip(boxes, track_ids, class_ids, confidences):
                x1, y1, x2, y2 = box
                cls_name = TRAFFIC_CLASSES.get(cls_id, "Vehicle")

                # Vehicle centroid (cx, cy)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Track vertical movement history
                if track_id not in track_positions:
                    track_positions[track_id] = cy
                prev_cy = track_positions[track_id]
                track_positions[track_id] = cy

                # Line crossing check: moving top-to-bottom across line_y
                if prev_cy < line_y <= cy and track_id not in counted_ids:
                    counted_ids.add(track_id)
                    cumulative_counts[cls_name] = cumulative_counts.get(cls_name, 0) + 1
                    # Flash tripwire green when crossing occurs
                    cv2.line(frame, (0, line_y), (width, line_y), (0, 255, 0), 4)

                # Color: Green if already counted, Orange/Cyan if waiting
                box_color = (0, 255, 0) if track_id in counted_ids else (255, 150, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

                label = f"ID:{track_id} {cls_name} {int(conf * 100)}%"
                cv2.putText(frame, label, (x1, max(20, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 2)

        # Periodic dictionary cleanup to prevent unbounded RAM growth
        if len(track_positions) > 500:
            track_positions.clear()

        # Telemetry Display Box
        total_passed = sum(cumulative_counts.values())
        cv2.rectangle(frame, (10, 10), (350, 130), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (350, 130), (0, 255, 255), 1)

        cv2.putText(frame, "TRAFFIQ TRACKER (YOLOv8s + ByteTrack)", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(frame, f"Active In Scene: {current_active_density}", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)
        cv2.putText(frame, f"Total Vehicles Passed: {total_passed}", (20, 92), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.putText(frame, f"Cars: {cumulative_counts['Car']} | Bikes: {cumulative_counts['Motorcycle']} | Trucks: {cumulative_counts['Truck']}", 
                    (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

        cv2.imshow("TRAFFIQ - Vehicle Tracking & Tripwire Counting", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Tracking stream closed cleanly.")

if __name__ == "__main__":
    run_traffic_tracking(0)