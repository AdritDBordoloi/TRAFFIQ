"""Tier 2: Component Tests for Web Dashboard Backend & REST API.
Verifies Flask endpoints, video feed streaming, event filtering,
clean JSON serialization (zero NumPy types), and demo triggers.
"""

from __future__ import annotations

import collections
import os
import shutil
import tempfile
import unittest
from typing import Any, Dict, List

import numpy as np

from traffiq.database.connection import create_connection, register_sqlite_adapters
from traffiq.database.schema import init_schema
try:
    from traffiq.web.engine import DashboardEngine
    from traffiq.web.server import create_app
except (ImportError, ModuleNotFoundError, AttributeError):
    DashboardEngine = None
    create_app = None


class FakeDashboardEngine:
    """Mock DashboardEngine for isolated web API unit testing."""

    def __init__(self) -> None:
        self.stats = {
            "vehicles": 12,
            "violations": 3,
            "emergencies": 1,
            "total_fines": 3000,
            "fps": 28.5,
        }
        self.signal_state = "GREEN"
        self.time_remaining = 6.2
        self.is_emergency_locked = False
        self.events = collections.deque([
            {
                "id": 1,
                "type": "violation",
                "timestamp": "2026-10-07 10:00:00",
                "payload": {
                    "challan_id": 1,
                    "track_id": 801,
                    "plate": "AS01AB1234",
                    "owner": "Rajesh Sharma",
                    "fine": 1000,
                },
            },
            {
                "id": 2,
                "type": "emergency",
                "timestamp": "2026-10-07 10:05:00",
                "payload": {
                    "track_id": 991,
                    "vehicle_type": "Ambulance",
                    "status": "active",
                },
            },
        ], maxlen=100)

    def get_latest_frame(self) -> bytes:
        # Return valid minimal JPEG bytes
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        import cv2
        ret, buf = cv2.imencode(".jpg", img)
        return buf.tobytes() if ret else b""

    def get_state(self) -> Dict[str, Any]:
        return {
            "stats": {k: int(v) if isinstance(v, (int, np.integer)) else float(v) for k, v in self.stats.items()},
            "signal": {
                "state": self.signal_state,
                "time_remaining": float(self.time_remaining),
                "is_emergency_locked": bool(self.is_emergency_locked),
            },
            "emergency_active": bool(self.is_emergency_locked),
        }

    def get_events(self, since_id: int = 0) -> List[Dict[str, Any]]:
        return [e for e in self.events if e["id"] > since_id]

    def toggle_signal(self) -> Dict[str, Any]:
        if self.is_emergency_locked:
            return {"success": False, "message": "Signal locked"}
        self.signal_state = "RED" if self.signal_state == "GREEN" else "GREEN"
        return {"success": True, "state": self.signal_state}

    def trigger_demo_violation(self) -> Dict[str, Any]:
        ev = {
            "id": len(self.events) + 1,
            "type": "violation",
            "timestamp": "2026-10-07 10:10:00",
            "payload": {
                "challan_id": 99,
                "track_id": 899,
                "plate": "DL04CA9999",
                "owner": "Priya Verma",
                "fine": 1000,
            },
        }
        self.events.append(ev)
        self.stats["violations"] += 1
        self.stats["total_fines"] += 1000
        return ev

    def trigger_demo_ambulance(self, duration: float = 15.0) -> Dict[str, Any]:
        self.is_emergency_locked = True
        ev = {
            "id": len(self.events) + 1,
            "type": "emergency",
            "timestamp": "2026-10-07 10:12:00",
            "payload": {
                "vehicle_type": "Ambulance",
                "status": "active",
                "duration": float(duration),
            },
        }
        self.events.append(ev)
        self.stats["emergencies"] += 1
        return ev

    def reset_stats(self) -> Dict[str, Any]:
        self.stats["vehicles"] = 0
        self.stats["violations"] = 0
        self.stats["emergencies"] = 0
        self.stats["total_fines"] = 0
        return {"success": True}


class TestWebAPI(unittest.TestCase):
    """Component tests verifying Flask REST API endpoints and data formats."""

    def setUp(self) -> None:
        if create_app is None or DashboardEngine is None:
            self.skipTest('web backend not yet implemented')
        register_sqlite_adapters()
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test_web.db")
        self.violations_dir = os.path.join(self.test_dir, "violations")
        os.makedirs(self.violations_dir, exist_ok=True)

        # Initialize SQLite test database
        conn = create_connection(self.db_path)
        init_schema(conn)
        # Insert a sample challan
        conn.execute("""
            INSERT INTO challan_records (
                timestamp, track_id, plate_number, owner_name, violation_type, fine_amount, evidence_path
            ) VALUES ('2026-10-07 10:00:00', 42, 'AS01AB1234', 'Rajesh Sharma', 'Red Light Jump', 1000, '')
        """)
        conn.commit()
        conn.close()

        self.fake_engine = FakeDashboardEngine()
        self.app = create_app(
            engine=self.fake_engine,
            violations_dir=self.violations_dir,
            db_path=self.db_path,
        )
        self.client = self.app.test_client()

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_index_page_returns_html(self) -> None:
        """GET / returns 200 OK and serves index.html."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("TRAFFIQ", res.get_data(as_text=True))

    def test_static_assets_served(self) -> None:
        """GET /style.css and /app.js return 200 OK."""
        res_css = self.client.get("/style.css")
        self.assertEqual(res_css.status_code, 200)
        res_js = self.client.get("/app.js")
        self.assertEqual(res_js.status_code, 200)

    def test_get_state_returns_valid_json(self) -> None:
        """GET /api/state returns valid JSON with stats, signal, and emergency keys."""
        res = self.client.get("/api/state")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("stats", data)
        self.assertIn("signal", data)
        self.assertIn("emergency_active", data)
        self.assertEqual(data["stats"]["vehicles"], 12)
        self.assertEqual(data["signal"]["state"], "GREEN")

    def test_json_has_no_numpy_types(self) -> None:
        """Ensure all fields in JSON responses are native Python types, not NumPy types."""
        res = self.client.get("/api/state")
        data = res.get_json()
        
        def check_no_numpy(val: Any) -> None:
            self.assertFalse(isinstance(val, (np.integer, np.floating)), f"Found NumPy scalar: {val}")
            if isinstance(val, dict):
                for v in val.values():
                    check_no_numpy(v)
            elif isinstance(val, list):
                for v in val:
                    check_no_numpy(v)

        check_no_numpy(data)

    def test_get_events_since_filter(self) -> None:
        """GET /api/events respects ?since=ID query parameter."""
        res1 = self.client.get("/api/events?since=0")
        self.assertEqual(res1.status_code, 200)
        events1 = res1.get_json()
        self.assertEqual(len(events1), 2)

        res2 = self.client.get("/api/events?since=1")
        events2 = res2.get_json()
        self.assertEqual(len(events2), 1)
        self.assertEqual(events2[0]["id"], 2)

    def test_get_challans_joins_and_limits(self) -> None:
        """GET /api/challans returns list of records with native int track_id."""
        res = self.client.get("/api/challans?limit=5")
        self.assertEqual(res.status_code, 200)
        challans = res.get_json()
        self.assertIsInstance(challans, list)
        self.assertGreaterEqual(len(challans), 1)
        first = challans[0]
        self.assertIsInstance(first["track_id"], int)
        self.assertIsInstance(first["fine_amount"], int)
        self.assertEqual(first["plate_number"], "AS01AB1234")

    def test_challans_limit_is_validated_and_bounded(self) -> None:
        """Junk / non-positive limits must not dump the table (SQLite treats LIMIT -1 as unlimited)."""
        conn = create_connection(self.db_path)
        for i in range(3):
            conn.execute(
                "INSERT INTO challan_records (timestamp, track_id, plate_number, owner_name, "
                "violation_type, fine_amount, evidence_path) VALUES "
                "('2026-10-07 11:00:00', ?, 'KA05MJ8821', 'Vikram Rao', 'Red Light Jump', 1000, '')",
                (100 + i,),
            )
        conn.commit()
        conn.close()

        self.assertEqual(len(self.client.get("/api/challans?limit=-1").get_json()), 1)
        self.assertEqual(len(self.client.get("/api/challans?limit=0").get_json()), 1)
        self.assertEqual(len(self.client.get("/api/challans?limit=2").get_json()), 2)
        self.assertEqual(len(self.client.get("/api/challans?limit=abc").get_json()), 4)

    def test_json_keys_keep_insertion_order(self) -> None:
        """Flask 3 ignores config['JSON_SORT_KEYS']; the provider flag must be disabled."""
        self.assertFalse(self.app.json.sort_keys)

    def test_demo_ambulance_sanitises_duration(self) -> None:
        """Non-numeric / absurd durations fall back to a safe, bounded value instead of a 500."""
        res = self.client.post("/api/demo/ambulance", json={"duration": "soon"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json()["event"]["payload"]["duration"], 15.0)

        res = self.client.post("/api/demo/ambulance", json={"duration": 1e9})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json()["event"]["payload"]["duration"], 120.0)

    def test_toggle_signal_endpoint(self) -> None:
        """POST /api/signal/toggle flips signal state."""
        res = self.client.get("/api/state")
        old_state = res.get_json()["signal"]["state"]

        res_post = self.client.post("/api/signal/toggle")
        self.assertEqual(res_post.status_code, 200)
        data = res_post.get_json()
        self.assertTrue(data["success"])
        self.assertNotEqual(data["state"], old_state)

    def test_demo_violation_endpoint(self) -> None:
        """POST /api/demo/violation creates a simulated violation event."""
        res = self.client.post("/api/demo/violation")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["event"]["type"], "violation")

    def test_demo_ambulance_endpoint(self) -> None:
        """POST /api/demo/ambulance triggers emergency mode."""
        res = self.client.post("/api/demo/ambulance", json={"duration": 10.0})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["event"]["type"], "emergency")

        # Verify state is locked
        res_state = self.client.get("/api/state")
        self.assertTrue(res_state.get_json()["emergency_active"])

    def test_reset_stats_endpoint(self) -> None:
        """POST /api/reset resets live counters."""
        res = self.client.post("/api/reset")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.get_json()["success"])

    def test_video_feed_header(self) -> None:
        """GET /video_feed returns multipart stream content type."""
        res = self.client.get("/video_feed")
        self.assertEqual(res.status_code, 200)
        self.assertIn("multipart/x-mixed-replace", res.headers.get("Content-Type", ""))


if __name__ == "__main__":
    unittest.main()
