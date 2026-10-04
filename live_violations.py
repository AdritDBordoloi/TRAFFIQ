import cv2
import time
import os
import sys
from datetime import datetime
from ultralytics import YOLO

TRAFFIC_CLASSES = {
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

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

                        # Save evidence crop with timestamp
                        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                        crop_y1 = max(0, y1 - 10)
                        crop_y2 = min(height, y2 + 10)
                        crop_x1 = max(0, x1 - 10)
                        crop_x2 = min(width, x2 + 10)
                        evidence_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                        filename = f"violations/violation_ID{track_id}_{timestamp_str}.jpg"
                        cv2.imwrite(filename, evidence_crop)
                        print(f"[ALERT] Red-Light Violation logged for ID:{track_id}! Evidence: {filename}")

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