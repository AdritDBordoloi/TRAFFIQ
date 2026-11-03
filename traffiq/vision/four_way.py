"""
4-Way Intersection Multi-Stream & Quadrant Video Processor for TRAFFIQ.
=======================================================================
Processes 4-way intersection video inputs:
1. Single 4-way video clip (2x2 quadrant split or overhead crossroad view).
2. Four separate video clips or camera streams (North, South, East, West).
3. Dynamic synthetic 4-way intersection simulation when video feeds are inactive.

Integrates with YOLOv8 to detect vehicle queues per approach and feeds real-time
counts directly into the AdaptiveIntersectionController.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from traffiq.enforcement.adaptive_signal import AdaptiveIntersectionController, Approach, SignalColor
from traffiq.vision.detector import VehicleDetector


class FourWayVideoProcessor:
    """
    Manages video capture, 4-quadrant composition, YOLO vehicle detection,
    and visual HUD annotation for 4-way intersection approaches.
    """

    def __init__(
        self,
        source_4way: Optional[str] = None,
        sources_dict: Optional[Dict[str, str]] = None,
        mode: str = "quadrant",
        detector: Optional[VehicleDetector] = None,
        device: str = "0",
    ) -> None:
        """
        Initialize the 4-way video processor.

        Args:
            source_4way: Path to single 4-way video file or camera index.
            sources_dict: Map of approach names ('NORTH', 'SOUTH', 'EAST', 'WEST') to video paths.
            mode: 'quadrant' (2x2 grid), 'crossroad' (overhead), or 'multi' (4 separate streams).
            detector: Optional shared VehicleDetector instance.
            device: Inference device ('0' or 'cpu').
        """
        self.mode = mode if mode in ("quadrant", "crossroad", "multi") else "quadrant"
        self.device = device
        self.detector = detector

        self._caps: Dict[str, cv2.VideoCapture] = {}
        self._single_cap: Optional[cv2.VideoCapture] = None
        self._is_video_file: bool = False
        self._detect_error_reported = False

        self.last_detected_counts: Dict[str, int] = {
            app: 0 for app in Approach.all_approaches()
        }
        self.last_detected_emergencies: Dict[str, bool] = {
            app: False for app in Approach.all_approaches()
        }

        if source_4way or sources_dict:
            self.set_sources(source_4way=source_4way, sources_dict=sources_dict, mode=self.mode)

    def set_sources(
        self,
        source_4way: Optional[str] = None,
        sources_dict: Optional[Dict[str, str]] = None,
        mode: Optional[str] = None,
    ) -> bool:
        """
        Configure or switch video sources at runtime.
        """
        self.close()
        if mode:
            self.mode = mode

        if sources_dict and len(sources_dict) > 0:
            self.mode = "multi"
            for app in Approach.all_approaches():
                src = str(sources_dict.get(app, "")).strip("'\"")
                if src:
                    try:
                        if src.isdigit():
                            cap = cv2.VideoCapture(int(src), cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
                        else:
                            cap = cv2.VideoCapture(src)
                        if cap and cap.isOpened():
                            self._caps[app] = cap
                    except Exception as e:
                        print(f"[WARN] Failed opening source for {app}: {e}")
            return len(self._caps) > 0

        elif source_4way:
            src_str = str(source_4way).strip("'\"")
            try:
                if src_str.isdigit():
                    self._single_cap = cv2.VideoCapture(int(src_str), cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
                    self._is_video_file = False
                elif os.path.exists(src_str):
                    self._single_cap = cv2.VideoCapture(src_str)
                    self._is_video_file = True
                if self._single_cap and self._single_cap.isOpened():
                    return True
            except Exception as e:
                print(f"[WARN] Failed opening 4-way source {src_str}: {e}")

        return False

    def close(self) -> None:
        """Safely release all open video captures."""
        if self._single_cap is not None:
            try:
                self._single_cap.release()
            except Exception:
                pass
            self._single_cap = None

        for cap in self._caps.values():
            try:
                cap.release()
            except Exception:
                pass
        self._caps.clear()

    def __del__(self) -> None:
        self.close()

    def read_raw_frame(self) -> Tuple[bool, np.ndarray]:
        """
        Fetch the next composite 1280x720 frame from the active sources,
        or generate an animated simulation frame if no source is active.
        """
        # Mode A: 4 Separate video captures
        if self.mode == "multi" and len(self._caps) > 0:
            quad_frames = {}
            for app in Approach.all_approaches():
                cap = self._caps.get(app)
                f = None
                if cap is not None and cap.isOpened():
                    ret, f = cap.read()
                    if not ret or f is None:
                        # Loop video
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, f = cap.read()
                if f is None:
                    f = self._create_sub_placeholder(app)
                f_resized = cv2.resize(f, (640, 360), interpolation=cv2.INTER_LINEAR)
                quad_frames[app] = f_resized

            # Stitch into 1280x720:
            # Top-Left: NORTH, Top-Right: EAST
            # Bottom-Left: WEST, Bottom-Right: SOUTH
            top_row = np.hstack([quad_frames[Approach.NORTH.value], quad_frames[Approach.EAST.value]])
            bottom_row = np.hstack([quad_frames[Approach.WEST.value], quad_frames[Approach.SOUTH.value]])
            canvas = np.vstack([top_row, bottom_row])
            return True, canvas

        # Mode B: Single 4-way video clip
        if self._single_cap is not None and self._single_cap.isOpened():
            ret, frame = self._single_cap.read()
            if not ret or frame is None:
                if self._is_video_file:
                    self._single_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self._single_cap.read()
            if ret and frame is not None:
                if frame.shape[1] != 1280 or frame.shape[0] != 720:
                    frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
                return True, frame

        # Fallback: Synthetic animated 4-way crossroad frame
        return False, self._create_synthetic_crossroad_frame()

    def process_and_annotate(
        self,
        frame: np.ndarray,
        controller: AdaptiveIntersectionController,
    ) -> Tuple[np.ndarray, Dict[str, int], Dict[str, bool]]:
        """
        Run detection and draw visual 4-way HUD onto frame.
        """
        h, w = frame.shape[:2]
        counts = {app: 0 for app in Approach.all_approaches()}
        emergencies = {app: False for app in Approach.all_approaches()}

        # Run YOLO detection if available
        if self.detector is not None:
            try:
                detections = self.detector.detect(frame)
                for det in detections:
                    cx, cy = det.centroid
                    app = self._assign_approach(cx, cy, w, h)
                    counts[app] += 1

                    # Draw vehicle box
                    x1, y1, x2, y2 = det.bbox
                    color = (56, 189, 248)  # Cyan
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"[{app}] {det.class_name} {int(det.confidence*100)}%",
                        (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        color,
                        1,
                    )
            except Exception as exc:  # keep the stream alive; report the first failure only
                if not self._detect_error_reported:
                    self._detect_error_reported = True
                    print(f"[WARN] 4-way vehicle detection failed: {exc}")

        # Draw 4-Quadrant Partition Grid & Directional HUD
        self._draw_quadrant_hud(frame, controller, counts)

        self.last_detected_counts = counts
        self.last_detected_emergencies = emergencies
        return frame, counts, emergencies

    def _assign_approach(self, cx: int, cy: int, w: int, h: int) -> str:
        """Map a 2D vehicle centroid to one of the 4 intersection approaches."""
        if self.mode == "quadrant" or self.mode == "multi":
            # 2x2 Quadrant split:
            # Top-Left = NORTH, Top-Right = EAST
            # Bottom-Left = WEST, Bottom-Right = SOUTH
            if cx < w // 2 and cy < h // 2:
                return Approach.NORTH.value
            elif cx >= w // 2 and cy < h // 2:
                return Approach.EAST.value
            elif cx < w // 2 and cy >= h // 2:
                return Approach.WEST.value
            else:
                return Approach.SOUTH.value
        else:
            # Overhead crossroad mode (directional angle from center)
            mid_x, mid_y = w // 2, h // 2
            dx = cx - mid_x
            dy = cy - mid_y
            if abs(dx) > abs(dy):
                return Approach.EAST.value if dx > 0 else Approach.WEST.value
            else:
                return Approach.SOUTH.value if dy > 0 else Approach.NORTH.value

    def _draw_quadrant_hud(
        self,
        frame: np.ndarray,
        controller: AdaptiveIntersectionController,
        counts: Dict[str, int],
    ) -> None:
        """Draw quadrant borders, signal lamps, and telemetry banners."""
        h, w = frame.shape[:2]
        half_w = w // 2
        half_h = h // 2

        # Grid separator lines
        cv2.line(frame, (half_w, 0), (half_w, h), (70, 80, 100), 2)
        cv2.line(frame, (0, half_h), (w, half_h), (70, 80, 100), 2)

        state = controller.get_state()
        time_rem = state.get("time_remaining", 0.0)
        active_app = state.get("active_approach", Approach.NORTH.value)

        quad_coords = {
            Approach.NORTH.value: (0, 0, half_w, half_h),
            Approach.EAST.value: (half_w, 0, w, half_h),
            Approach.WEST.value: (0, half_h, half_w, h),
            Approach.SOUTH.value: (half_w, half_h, w, h),
        }

        color_bgr_map = {
            SignalColor.GREEN.value: (34, 197, 94),     # Emerald Green
            SignalColor.YELLOW.value: (11, 158, 245),   # Amber/Yellow
            SignalColor.RED.value: (68, 68, 239),       # Red
        }

        for app, (x1, y1, x2, y2) in quad_coords.items():
            app_data = controller.approaches.get(app)
            light_state = app_data.state if app_data else SignalColor.RED.value
            bgr = color_bgr_map.get(light_state, (68, 68, 239))

            # Active arm glowing border
            if light_state in (SignalColor.GREEN.value, SignalColor.YELLOW.value):
                cv2.rectangle(frame, (x1 + 2, y1 + 2), (x2 - 2, y2 - 2), bgr, 3)

            # Header Banner
            cv2.rectangle(frame, (x1 + 10, y1 + 10), (x1 + 260, y1 + 45), (15, 23, 42), -1)
            cv2.rectangle(frame, (x1 + 10, y1 + 10), (x1 + 260, y1 + 45), bgr, 1)

            # Signal status lamp circle
            cv2.circle(frame, (x1 + 25, y1 + 27), 8, bgr, -1)
            cv2.putText(
                frame,
                f"{app} APPROACH",
                (x1 + 42, y1 + 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            # Telemetry Pill in Quadrant
            q_cnt = counts.get(app, 0)
            timer_text = f"{time_rem:.1f}s" if app == active_app else f"{int(app_data.wait_time if app_data else 0)}s wait"
            cv2.putText(
                frame,
                f"Queue: {q_cnt} cars | {timer_text}",
                (x1 + 15, y2 - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 220, 240),
                1,
            )

        # Center Arbiter Badge
        cx, cy = half_w, half_h
        cv2.rectangle(frame, (cx - 130, cy - 35), (cx + 130, cy + 35), (15, 23, 42), -1)
        cv2.rectangle(frame, (cx - 130, cy - 35), (cx + 130, cy + 35), (56, 189, 248), 2)
        cv2.putText(
            frame,
            f"ARBITER: {active_app}",
            (cx - 110, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (56, 189, 248),
            2,
        )
        cv2.putText(
            frame,
            f"Phase: {controller.phase_state} ({time_rem:.1f}s)",
            (cx - 100, cy + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    def _create_sub_placeholder(self, approach_name: str) -> np.ndarray:
        """Create a standby quadrant image."""
        img = np.full((360, 640, 3), 30, dtype=np.uint8)
        cv2.rectangle(img, (20, 20), (620, 340), (50, 60, 80), 1)
        cv2.putText(
            img,
            f"{approach_name} CAM STREAM",
            (180, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (140, 160, 190),
            2,
        )
        cv2.putText(
            img,
            "Standby / Ready",
            (240, 215),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (80, 180, 100),
            1,
        )
        return img

    def _create_synthetic_crossroad_frame(self) -> np.ndarray:
        """Create a synthetic 4-way crossroad animation frame."""
        canvas = np.full((720, 1280, 3), 20, dtype=np.uint8)

        # Road asphalt cross
        # Vertical road (North-South)
        cv2.rectangle(canvas, (490, 0), (790, 720), (40, 45, 50), -1)
        # Horizontal road (West-East)
        cv2.rectangle(canvas, (0, 210), (1280, 510), (40, 45, 50), -1)

        # Lane markings (dashed lines)
        # Vertical center divider
        for y in range(0, 720, 35):
            if 210 < y < 510:
                continue
            cv2.line(canvas, (640, y), (640, y + 20), (200, 200, 200), 2)
        # Horizontal center divider
        for x in range(0, 1280, 35):
            if 490 < x < 790:
                continue
            cv2.line(canvas, (x, 360), (x + 20, 360), (200, 200, 200), 2)

        # Zebra Crossings
        for i in range(500, 780, 20):
            cv2.rectangle(canvas, (i, 195), (i + 12, 210), (230, 230, 230), -1)  # North
            cv2.rectangle(canvas, (i, 510), (i + 12, 525), (230, 230, 230), -1)  # South
        for j in range(220, 500, 20):
            cv2.rectangle(canvas, (475, j), (490, j + 12), (230, 230, 230), -1)  # West
            cv2.rectangle(canvas, (790, j), (805, j + 12), (230, 230, 230), -1)  # East

        return canvas
