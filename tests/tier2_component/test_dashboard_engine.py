"""Tier 2: DashboardEngine live-path regression tests.

Uses the *real* DashboardEngine (no capture thread, no model weights, no camera) to verify
that live detections are recorded as genuine challans / emergencies rather than demo data.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
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

    def test_demo_ambulance_is_labelled_as_demo(self) -> None:
        self.engine.trigger_demo_ambulance(duration=5.0)
        rows = self._rows("SELECT vehicle_type FROM emergency_events")
        self.assertEqual(rows, [("Ambulance (Demo)",)])

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
