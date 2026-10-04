import cv2
import time
import sys
from ultralytics import YOLO

# COCO dataset class IDs relevant to road traffic
TRAFFIC_CLASSES = {
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

def run_traffic_detection(camera_index=0):
    print("[INFO] Initializing YOLOv8n model...")
    # Loading pretrained weights (auto-downloads if not found locally)
    model = YOLO("yolov8s.pt")

    print(f"[INFO] Opening camera stream (Index: {camera_index})...")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {camera_index}.")
        return

    # Frame timing for FPS calculation
    prev_time = time.time()

    print("[SUCCESS] Pipeline active. Press 'q' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera stream interrupted.")
            break

        # Calculate live FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time

        # Run inference: filter strictly for traffic classes, confidence threshold >= 35%
        results = model.predict(
            source=frame,
            classes=list(TRAFFIC_CLASSES.keys()),
            conf=0.25,
            imgsz=640,
            device=0,  # 0 targets your RTX 4060 GPU
            verbose=False
        )[0]

        # Vehicle counter for the current frame
        counts = {"Car": 0, "Motorcycle": 0, "Bus": 0, "Truck": 0, "Bicycle": 0}

        # Parse detected bounding boxes
        for box in results.boxes:
            cls_id = int(box.cls[0])
            label_name = TRAFFIC_CLASSES.get(cls_id, "Vehicle")
            conf = float(box.conf[0])
            counts[label_name] = counts.get(label_name, 0) + 1

            # Get box coordinates [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw green bounding box around detected vehicle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw tag with class label and confidence score
            tag = f"{label_name} {int(conf * 100)}%"
            cv2.putText(
                frame, tag, 
                (x1, max(20, y1 - 8)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (0, 255, 0), 2
            )

        # Build telemetry banner overlay
        total_vehicles = sum(counts.values())
        cv2.rectangle(frame, (10, 10), (320, 110), (20, 20, 20), -1)
        cv2.rectangle(frame, (10, 10), (320, 110), (0, 255, 0), 1)

        cv2.putText(frame, f"TRAFFIQ DETECTOR (GPU)", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Live Vehicle Density: {total_vehicles}", (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"Cars: {counts['Car']} | Bikes: {counts['Motorcycle']} | Trucks/Buses: {counts['Truck'] + counts['Bus']}", 
                    (20, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        cv2.imshow("TRAFFIQ - Real-Time Vehicle Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Detection stream closed cleanly.")

if __name__ == "__main__":
    run_traffic_detection(0)