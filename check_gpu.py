import torch

print("=" * 45)
print("       TRAFFIQ HARDWARE ACCELERATION CHECK   ")
print("=" * 45)
cuda_available = torch.cuda.is_available()
print(f"CUDA Available : {cuda_available}")

if cuda_available:
    print(f"Target GPU     : {torch.cuda.get_device_name(0)}")
    print(f"VRAM Available : {round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)} GB")
    print("[STATUS] Ready for real-time high-speed inference.")
else:
    print("[WARNING] Running on CPU. Detections will be significantly slower.")
print("=" * 45)