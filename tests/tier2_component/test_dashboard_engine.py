"""Tier 2: DashboardEngine live-path regression tests.

Uses the *real* DashboardEngine (no capture thread, no model weights, no camera) to verify
that live detections are recorded as genuine challans / emergencies rather than demo data.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
import time
import unittest

import cv2
import numpy as np

try:
    from traffiq.web.engine import DashboardEngine
except (ImportError, ModuleNotFoundError, AttributeError):
    DashboardEngine = None


class TestDashboardEngineLivePath(unittest.TestCase):
    def setUp(self) -> None:
        if DashboardEngine is None:
            self.skipTest("web backend not available")
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "engine.db")
        self.vdir = os.path.join(self.tmp, "violations")
        self.engine = DashboardEngine(
            db_path=self.db,
            violations_dir=self.vdir,
            device="cpu",
            fine_amount=1500,
            start_thread=False,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _rows(self, sql: str, params: tuple = ()) -> list:
        conn = sqlite3.connect(self.db)
        try:
            return conn.execute(sql, params).fetchall()
        finally:
            conn.close()

    # ------------------------------------------------------------------ enforcement rules
    def test_identified_ambulance_is_exempt_from_red_light_enforcement(self) -> None:
        """Parity with live_unified.py: emergency vehicles are never ticketed for crossing on RED."""
        from traffiq.web.engine import is_enforceable_crossing

        self.assertTrue(is_enforceable_crossing(True, "RED", False))
        self.assertFalse(is_enforceable_crossing(True, "RED", True))
        self.assertFalse(is_enforceable_crossing(True, "GREEN", False))
        self.assertFalse(is_enforceable_crossing(False, "RED", False))

    # ------------------------------------------------------------------ device selection
    def test_cuda_device_string_maps_to_valid_torch_device(self) -> None:
        """Regression: torch.device("0") raises, so run_dashboard crashed on every CUDA machine."""
        import torch
        from unittest import mock

        with mock.patch("torch.cuda.is_available", return_value=True):
            eng = DashboardEngine(
                db_path=self.db, violations_dir=self.vdir, device="0", start_thread=False
            )
        self.assertEqual(eng.device_str, "0")
        self.assertEqual(eng.device, torch.device("cuda:0"))

    def test_cpu_fallback_when_cuda_unavailable(self) -> None:
        import torch
        from unittest import mock

        with mock.patch("torch.cuda.is_available", return_value=False):
            eng = DashboardEngine(
                db_path=self.db, violations_dir=self.vdir, device="0", start_thread=False
            )
        self.assertEqual(eng.device_str, "cpu")
        self.assertEqual(eng.device, torch.device("cpu"))

    # ------------------------------------------------------------------ violations
    def test_live_violation_stores_real_evidence_and_track_id(self) -> None:
        crop = np.full((120, 200, 3), 77, dtype=np.uint8)
        ev = self.engine.record_violation(np.int64(37), "UNKNOWN", crop)

        payload = ev["payload"]
        self.assertEqual(ev["type"], "violation")
        self.assertEqual(payload["track_id"], 37)
        self.assertEqual(payload["plate"], "UNKNOWN")
        self.assertEqual(payload["owner"], "UNKNOWN / OUT-OF-STATE")
        self.assertEqual(payload["fine"], 1500)

        # evidence is the real crop, served from /evidence/<path under violations dir>
        self.assertRegex(payload["image_url"], r"^/evidence/violation_ID37_\d{8}_\d{6}\.jpg$")
        self.assertNotIn("..", payload["image_url"])
        saved = os.path.join(self.vdir, payload["image_url"][len("/evidence/"):])
        self.assertTrue(os.path.isfile(saved))
        self.assertEqual(cv2.imread(saved).shape, crop.shape)

        row = self._rows(
            "SELECT typeof(track_id), track_id, plate_number, violation_type FROM challan_records"
        )[0]
        self.assertEqual(row, ("integer", 37, "UNKNOWN", "Red Light Jump"))
        self.assertEqual(self.engine.stats["violations"], 1)
        self.assertEqual(self.engine.stats["total_fines"], 1500)

    def test_evidence_url_resolves_short_names_and_rejects_outside_paths(self) -> None:
        """Regression: realpath'd evidence vs unresolved dir (symlinks / 8.3 names) gave ../.. URLs."""
        from traffiq.web.engine import evidence_url

        inside = os.path.join(self.vdir, "a", "b.jpg")
        self.assertEqual(evidence_url(inside, self.vdir), "/evidence/a/b.jpg")
        self.assertEqual(evidence_url(os.path.realpath(inside), self.vdir), "/evidence/a/b.jpg")
        self.assertEqual(evidence_url(os.path.join(self.tmp, "elsewhere.jpg"), self.vdir), "")
        self.assertEqual(evidence_url("", self.vdir), "")

    def test_unreadable_plate_is_never_charged_to_a_sample_owner(self) -> None:
        crop = np.full((60, 90, 3), 10, dtype=np.uint8)
        self.engine.record_violation(5, "", crop)
        plates = [r[0] for r in self._rows("SELECT plate_number FROM challan_records")]
        self.assertEqual(plates, ["UNKNOWN"])

    def test_registered_plate_resolves_real_owner(self) -> None:
        crop = np.full((60, 90, 3), 10, dtype=np.uint8)
        ev = self.engine.record_violation(8, "dl04ca9999", crop)
        self.assertEqual(ev["payload"]["owner"], "Priya Verma")
        self.assertEqual(ev["payload"]["plate"], "DL04CA9999")

    def test_demo_violation_is_labelled_as_demo(self) -> None:
        self.engine.trigger_demo_violation()
        types = [r[0] for r in self._rows("SELECT violation_type FROM challan_records")]
        self.assertEqual(len(types), 1)
        self.assertTrue(types[0].endswith("(Demo)"), types[0])

    # ------------------------------------------------------------------ emergencies
    def test_live_emergency_uses_real_track_id(self) -> None:
        ev = self.engine._start_emergency(20.0, 123, "Ambulance")
        self.assertEqual(self.engine.emergency_vehicle_track_id, 123)
        self.assertTrue(self.engine.is_emergency_locked)
        self.assertEqual(ev["payload"]["track_id"], 123)
        rows = self._rows("SELECT track_id, vehicle_type, status FROM emergency_events")
        self.assertEqual(rows, [(123, "Ambulance", "active")])

    def test_state_names_the_ambulance_only_while_the_corridor_is_active(self) -> None:
        self.assertIsNone(self.engine.get_state()["signal"]["emergency_track_id"])
        self.engine._start_emergency(20.0, 123, "Ambulance")
        self.assertEqual(self.engine.get_state()["signal"]["emergency_track_id"], 123)
        self.engine.is_emergency_locked = False
        self.assertIsNone(self.engine.get_state()["signal"]["emergency_track_id"])

    def test_demo_ambulance_is_labelled_as_demo(self) -> None:
        self.engine.trigger_demo_ambulance(duration=5.0)
        rows = self._rows("SELECT vehicle_type FROM emergency_events")
        self.assertEqual(rows, [("Ambulance (Demo)",)])

    # ------------------------------------------------------------------ video source
    def _write_video(self, name: str, gray: int) -> str:
        path = os.path.join(self.tmp, name)
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (640, 400))
        for _ in range(10):
            writer.write(np.full((400, 640, 3), gray, dtype=np.uint8))
        writer.release()
        return path

    def _running_engine(self, source: str) -> DashboardEngine:
        engine = DashboardEngine(
            source=source,
            model_path=os.path.join(self.tmp, "absent.pt"),
            oiv7_path=os.path.join(self.tmp, "absent.pt"),
            db_path=self.db,
            violations_dir=self.vdir,
            device="cpu",
            start_thread=True,
        )
        self.addCleanup(engine.stop)
        return engine

    @staticmethod
    def _wait_for_frame(engine: DashboardEngine, predicate, timeout: float = 15.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            frame = cv2.imdecode(np.frombuffer(engine.get_latest_frame(), np.uint8), cv2.IMREAD_COLOR)
            if frame is not None and predicate(frame):
                return frame
            time.sleep(0.05)
        raise AssertionError("expected frame never appeared")

    def test_stream_draws_stop_line_and_exit_line(self) -> None:
        """Both lines from live_unified.py must be on the frames sent to the browser."""
        engine = self._running_engine(self._write_video("a.mp4", 40))
        frame = self._wait_for_frame(engine, lambda f: f.shape[:2] == (400, 640) and int(f[390, 600, 0]) < 90)

        stop_b, stop_g, stop_r = (int(v) for v in frame[int(400 * 0.55), 600])
        self.assertTrue(stop_g > 150 or stop_r > 150, (stop_b, stop_g, stop_r))  # green or red signal
        exit_b, exit_g, exit_r = (int(v) for v in frame[int(400 * 0.72), 600])
        self.assertLess(exit_b, 110)  # yellow in BGR: low blue, high green and red
        self.assertGreater(exit_g, 150)
        self.assertGreater(exit_r, 150)

    def test_set_source_switches_the_live_stream(self) -> None:
        engine = self._running_engine(self._write_video("dark.mp4", 40))
        self._wait_for_frame(engine, lambda f: f.shape[:2] == (400, 640) and int(f[390, 600, 0]) < 90)

        bright = self._write_video("bright.mp4", 200)
        self.assertEqual(engine.set_source(bright), {"kind": "video", "label": "bright.mp4"})

        self._wait_for_frame(engine, lambda f: f.shape[:2] == (400, 640) and int(f[390, 600, 0]) > 150)
        self.assertEqual(engine.get_state()["source"]["label"], "bright.mp4")

    def test_slow_detection_does_not_slow_the_video(self) -> None:
        """Regression: detection shared a thread with decoding, so the stream ran at inference speed."""
        engine = DashboardEngine(
            source=self._write_video("steady.mp4", 40),
            model_path=os.path.join(self.tmp, "absent.pt"),
            oiv7_path=os.path.join(self.tmp, "absent.pt"),
            db_path=self.db,
            violations_dir=self.vdir,
            device="cpu",
            start_thread=False,
        )

        class SlowTracker:
            calls = 0

            def track(self, frame, frame_idx=None):
                type(self).calls += 1
                time.sleep(0.25)
                return []

        engine.tracker = SlowTracker()
        engine.start()
        self.addCleanup(engine.stop)
        self._wait_for_frame(engine, lambda f: f.shape[:2] == (400, 640))

        start_seq, _ = engine.wait_for_frame(0)
        start_calls = SlowTracker.calls
        time.sleep(1.0)
        end_seq, _ = engine.wait_for_frame(0)

        self.assertLessEqual(SlowTracker.calls - start_calls, 5)
        self.assertGreaterEqual(end_seq - start_seq, 12)  # the clip plays at 20 FPS

    def test_boxes_stay_on_a_moving_vehicle_despite_slow_detection(self) -> None:
        """Regression: boxes from an older analysed frame were drawn on the newest video frame, so they trailed."""
        from traffiq.vision.tracker import TrackedVehicle

        speed, top, bottom, half_width = 8, 170, 235, 30
        path = os.path.join(self.tmp, "moving.mp4")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 400))
        for i in range(60):
            image = np.zeros((400, 640, 3), dtype=np.uint8)
            cx = 60 + i * speed
            image[top:bottom, cx - half_width:cx + half_width] = 255
            writer.write(image)
        writer.release()

        class FindBlock:
            """Stands in for a detector that takes as long as real inference does under load."""

            def track(self, frame, frame_idx=None):
                time.sleep(0.08)
                xs = np.where(frame[(top + bottom) // 2, :, 0] > 200)[0]
                if xs.size == 0:
                    return []
                x1, x2 = int(xs[0]), int(xs[-1])
                return [TrackedVehicle(1, (x1, top, x2, bottom), 7, "Truck", ((x1 + x2) // 2, (top + bottom) // 2), 0.9)]

        engine = DashboardEngine(
            source=path,
            model_path=os.path.join(self.tmp, "absent.pt"),
            oiv7_path=os.path.join(self.tmp, "absent.pt"),
            db_path=self.db,
            violations_dir=self.vdir,
            device="cpu",
            start_thread=False,
        )
        engine.tracker = FindBlock()
        engine.start()
        self.addCleanup(engine.stop)

        offsets = []
        seq = 0
        deadline = time.time() + 3.0
        row = top + 10  # above the stop line, and below the box label
        while time.time() < deadline:
            seq, jpeg = engine.wait_for_frame(seq, timeout=0.5)
            frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            pixels = frame[row].astype(int)
            block = np.where(pixels.min(axis=1) > 200)[0]
            # The box's sides are orange in BGR: strong blue, medium green, little red
            sides = np.where((pixels[:, 0] > 170) & (pixels[:, 1] > 90) & (pixels[:, 1] < 200) & (pixels[:, 2] < 90))[0]
            if block.size > 20 and sides.size >= 2 and 200 < block.mean() < 480:
                offsets.append(abs((int(sides[0]) + int(sides[-1])) / 2 - block.mean()))

        self.assertGreater(len(offsets), 20)
        self.assertLess(float(np.median(offsets)), speed)  # under one frame of motion

    def test_boxes_blend_between_the_detections_around_a_frame(self) -> None:
        from traffiq.vision.tracker import TrackedVehicle
        from traffiq.web.engine import DetectionOverlay, overlay_for_frame

        def car(track_id, x):
            return TrackedVehicle(track_id, (x, 0, x + 10, 10), 2, "Car", (x + 5, 5), 0.9)

        history = [
            DetectionOverlay(0, 10, [car(1, 100), car(2, 500)]),
            DetectionOverlay(0, 20, [car(1, 200), car(3, 700)]),
        ]
        x_of = lambda overlay: {v.track_id: v.bbox[0] for v in overlay.tracked}  # noqa: E731

        self.assertEqual(overlay_for_frame([], 5).tracked, [])
        self.assertEqual(x_of(overlay_for_frame(history, 10)), {1: 100, 2: 500})  # analysed: exact
        self.assertEqual(x_of(overlay_for_frame(history, 20)), {1: 200, 3: 700})  # analysed: exact
        self.assertEqual(x_of(overlay_for_frame(history, 12)), {1: 120, 2: 500})  # lost car stays for now
        self.assertEqual(x_of(overlay_for_frame(history, 18)), {1: 180, 3: 700})  # new car has appeared
        self.assertEqual(overlay_for_frame(history, 25).tracked[0].bbox[0], 250)  # continues at its speed
        self.assertEqual(overlay_for_frame(history, 500).tracked[0].bbox[0], 300)  # within a limit
        self.assertEqual(overlay_for_frame(history, 15).tracked[0].centroid, (155, 5))

    def test_slow_plate_reading_runs_outside_the_video_pipeline(self) -> None:
        crop = np.full((60, 90, 3), 10, dtype=np.uint8)
        self.engine.plate_extractor.read_plate = lambda image: ("DL04CA9999", 0.9)
        self.engine._violation_queue.put((12, crop))
        self.engine._running = False  # the worker finishes queued violations, then exits

        self.engine._violation_loop()

        self.assertEqual(self._rows("SELECT track_id, plate_number FROM challan_records"), [(12, "DL04CA9999")])

    def test_wait_for_frame_returns_only_when_a_newer_frame_exists(self) -> None:
        seq, _ = self.engine.wait_for_frame(0, timeout=0.01)
        started = time.time()
        same_seq, _ = self.engine.wait_for_frame(seq, timeout=0.05)
        self.assertEqual(same_seq, seq)
        self.assertGreaterEqual(time.time() - started, 0.04)

        self.engine._publish_jpeg(b"new")
        new_seq, jpeg = self.engine.wait_for_frame(seq, timeout=1.0)
        self.assertEqual((new_seq, jpeg), (seq + 1, b"new"))

    def test_video_pacing_waits_when_early_and_skips_when_late(self) -> None:
        from traffiq.web.engine import plan_video_frame

        self.assertEqual(plan_video_frame(10.0, 10.0, 0, 0.02), (0.0, 0))
        wait, skip = plan_video_frame(10.0, 10.0, 5, 0.02)
        self.assertAlmostEqual(wait, 0.1)
        self.assertEqual(skip, 0)
        self.assertEqual(plan_video_frame(10.5, 10.0, 5, 0.02), (0.0, 20))

    def test_codec_names_select_the_decoder(self) -> None:
        from traffiq.web.engine import SOFTWARE_DECODE_CODECS, fourcc_name

        self.assertEqual(fourcc_name(cv2.VideoWriter_fourcc(*"AV01")), "av01")
        self.assertIn(fourcc_name(cv2.VideoWriter_fourcc(*"h264")), SOFTWARE_DECODE_CODECS)
        self.assertNotIn(fourcc_name(cv2.VideoWriter_fourcc(*"AV01")), SOFTWARE_DECODE_CODECS)

    def test_rate_meter_reports_events_per_second(self) -> None:
        from traffiq.web.engine import RateMeter

        meter = RateMeter(window_seconds=1.0)
        rate = 0.0
        for i in range(200):
            rate = meter.tick(i * 0.02)
        self.assertAlmostEqual(rate, 50.0, delta=1.0)

    def test_missing_video_shows_placeholder_instead_of_opening_webcam(self) -> None:
        """Regression: a mistyped --source path silently fell back to the webcam."""
        cap, is_file = self.engine._open_capture(os.path.join(self.tmp, "typo.mp4"))
        self.assertFalse(is_file)
        self.assertFalse(cap.isOpened())

    def test_new_source_forgets_previous_tracks_but_keeps_challans(self) -> None:
        """Track ids restart per video, so a reused id must not suppress a new video's violations."""
        crop = np.full((60, 90, 3), 10, dtype=np.uint8)
        self.engine.record_violation(5, "UNKNOWN", crop)
        frame = np.zeros((400, 640, 3), dtype=np.uint8)
        event = self.engine.violation_detector.evaluate_crossing(5, True, "RED", (10, 10, 60, 60), frame)
        self.assertIsNotNone(event)
        self.engine.active_emergency_tracks.add(5)
        self.engine._seen_vehicle_ids.add(5)
        self.engine._start_emergency(20.0, 5, "Ambulance")

        self.engine._reset_for_new_source()

        self.assertIsNotNone(self.engine.violation_detector.evaluate_crossing(5, True, "RED", (10, 10, 60, 60), frame))
        self.assertNotIn(5, self.engine.active_emergency_tracks)
        self.assertNotIn(5, self.engine._seen_vehicle_ids)
        self.assertFalse(self.engine.is_emergency_locked)
        self.assertEqual(self.engine.stats["violations"], 0)
        self.assertEqual(len(self._rows("SELECT challan_id FROM challan_records")), 1)

    # ------------------------------------------------------------------ stats
    def test_reset_stats_forgets_seen_vehicle_ids(self) -> None:
        for vid in (1, 2, 3):
            self.engine._seen_vehicle_ids.add(vid)
        self.engine.stats["vehicles"] = 3

        self.engine.reset_stats()

        self.assertEqual(len(self.engine._seen_vehicle_ids), 0)
        self.assertEqual(self.engine.stats["vehicles"], 0)
        self.assertNotIn(1, self.engine._seen_vehicle_ids)


if __name__ == "__main__":
    unittest.main()
