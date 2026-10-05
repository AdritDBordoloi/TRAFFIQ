from ultralytics import YOLO

print("[INFO] Loading official Open Images V7 model (auto-downloads on first run)...")
model = YOLO("yolov8s-oiv7.pt")

# Search for the Ambulance class in the 601 model classes
ambulance_classes = {cls_id: name for cls_id, name in model.names.items() if "ambulance" in name.lower()}

print("=" * 50)
print(f"Ambulance Classes Found: {ambulance_classes}")
print(f"Total Model Classes    : {len(model.names)}")
print("=" * 50)