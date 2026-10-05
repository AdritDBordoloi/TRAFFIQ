import torch
import time
from ultralytics import YOLO

print("CUDA Status:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))

# Benchmark pure model speed on random data (bypassing webcam)
model = YOLO("yolov8s.pt")
dummy_frame = torch.zeros((1, 3, 640, 640)).to("cuda" if torch.cuda.is_available() else "cpu")

# Warmup
for _ in range(10):
    _ = model(dummy_frame, verbose=False)

# Measure 50 inferences
if torch.cuda.is_available():
    torch.cuda.synchronize()
start = time.time()
for _ in range(50):
    _ = model(dummy_frame, verbose=False)
if torch.cuda.is_available():
    torch.cuda.synchronize()
elapsed = time.time() - start

fps = 50.0 / max(1e-6, elapsed)
print(f"Pure AI Processing Speed: {fps:.1f} FPS ({elapsed/50*1000:.1f} ms per frame)")