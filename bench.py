"""
Performance benchmark suite for TRAFFIQ (bench.py).
Measures:
1. Pure AI inference speed (YOLOv8s on CUDA / CPU).
2. Camera & Video Stream reading throughput (CameraStream).
3. Integrated stream detection throughput (CameraStream + VehicleDetector).
"""

import os
import time
import torch
from ultralytics import YOLO

from traffiq.vision.camera import CameraStream
from traffiq.vision.detector import VehicleDetector


def run_benchmark(num_inferences: int = 50, stream_frames: int = 60) -> dict:
    results = {}
    is_cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if is_cuda else "CPU"

    print("============================================================")
    print(" TRAFFIQ PERFORMANCE BENCHMARK (bench.py)")
    print("============================================================")
    print(f"CUDA Status: {is_cuda}")
    if is_cuda:
        print(f"Device Name: {device_name}")

    # ------------------------------------------------------------------
    # 1. Pure AI Model Speed Benchmark (Phase 2 Baseline Protocol)
    # ------------------------------------------------------------------
    model = YOLO("yolov8s.pt")
    dummy_frame = torch.zeros((1, 3, 640, 640)).to("cuda" if is_cuda else "cpu")

    # Warmup
    for _ in range(10):
        _ = model(dummy_frame, verbose=False)

    if is_cuda:
        torch.cuda.synchronize()
        torch.backends.cudnn.benchmark = True
    start = time.time()
    with torch.inference_mode():
        for _ in range(num_inferences):
            _ = model(dummy_frame, verbose=False)
    if is_cuda:
        torch.cuda.synchronize()
    elapsed = time.time() - start

    ai_fps = num_inferences / max(1e-6, elapsed)
    ai_ms = (elapsed / num_inferences) * 1000
    results["ai_fps"] = ai_fps
    results["ai_ms"] = ai_ms
    print(f"Pure AI Processing Speed: {ai_fps:.1f} FPS ({ai_ms:.1f} ms per frame)")

    # ------------------------------------------------------------------
    # 2. Camera / Video Stream Throughput
    # ------------------------------------------------------------------
    video_source = "uploads/Transportation_Stockholm_Roads.mkv"
    if not os.path.exists(video_source):
        video_source = 0

    try:
        stream = CameraStream(source=video_source, auto_open=True, threaded=False)
        count = 0
        start_stream = time.time()
        for idx, frame in stream.frames():
            count += 1
            if count >= stream_frames:
                break
        elapsed_stream = time.time() - start_stream
        stream.release()

        stream_fps = count / max(1e-6, elapsed_stream)
        stream_ms = (elapsed_stream / max(1, count)) * 1000
        results["stream_fps"] = stream_fps
        results["stream_ms"] = stream_ms
        print(f"Camera Stream Read Speed (Direct): {stream_fps:.1f} FPS ({stream_ms:.1f} ms per frame)")
    except Exception as e:
        results["stream_fps"] = 0.0
        results["stream_ms"] = 0.0
        print(f"Camera Stream Read Speed: N/A ({e})")

    # 2b. Asynchronous Threaded Stream Throughput
    try:
        t_stream = CameraStream(source=video_source, auto_open=True, threaded=True)
        time.sleep(0.05)
        t_count = 0
        start_t = time.time()
        for _ in range(stream_frames):
            ret, frame = t_stream.read()
            if ret and frame is not None:
                t_count += 1
            time.sleep(0.001)
        elapsed_t = time.time() - start_t
        t_stream.release()

        t_fps = t_count / max(1e-6, elapsed_t)
        t_ms = (elapsed_t / max(1, t_count)) * 1000
        results["threaded_stream_fps"] = t_fps
        results["threaded_stream_ms"] = t_ms
        print(f"Camera Stream Read Speed (Threaded): {t_fps:.1f} FPS ({t_ms:.1f} ms per frame)")
    except Exception as e:
        results["threaded_stream_fps"] = 0.0
        results["threaded_stream_ms"] = 0.0
        print(f"Threaded Stream Read Speed: N/A ({e})")

    # ------------------------------------------------------------------
    # 3. Integrated Stream + Vision Pipeline Benchmark
    # ------------------------------------------------------------------
    try:
        detector = VehicleDetector(device="0" if is_cuda else "cpu")
        stream = CameraStream(source=video_source, auto_open=True)

        # Warmup
        for idx, frame in stream.frames():
            _ = detector.detect(frame)
            if idx >= 5:
                break
        stream.release()

        stream = CameraStream(source=video_source, auto_open=True)
        if is_cuda:
            torch.cuda.synchronize()
        start_pipe = time.time()
        pipe_count = 0
        for idx, frame in stream.frames():
            _ = detector.detect(frame)
            pipe_count += 1
            if pipe_count >= stream_frames:
                break
        if is_cuda:
            torch.cuda.synchronize()
        elapsed_pipe = time.time() - start_pipe
        stream.release()

        pipe_fps = pipe_count / max(1e-6, elapsed_pipe)
        pipe_ms = (elapsed_pipe / max(1, pipe_count)) * 1000
        results["pipeline_fps"] = pipe_fps
        results["pipeline_ms"] = pipe_ms
        print(f"Stream + Vision Pipeline: {pipe_fps:.1f} FPS ({pipe_ms:.1f} ms per frame)")
    except Exception as e:
        results["pipeline_fps"] = 0.0
        results["pipeline_ms"] = 0.0
        print(f"Stream + Vision Pipeline: N/A ({e})")

    print("============================================================")
    return results


if __name__ == "__main__":
    run_benchmark()
