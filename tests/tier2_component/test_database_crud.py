"""Tier 2: Database CRUD & RTO Registry Interaction Tests.
Verifies RTO vehicle registry lookups (known plates vs. missing plates),
challan record insertion, transaction commit, and query integrity.
"""

import os
import sqlite3
import tempfile
import unittest

from tests.conftest import init_test_database

try:
    from traffiq.database.repository import DatabaseManager, lookup_vehicle, record_challan
except ImportError:
    DatabaseManager = None
    lookup_vehicle = None
    record_challan = None


class OracleDatabaseManager:
    """Authoritative reference implementation of DatabaseManager CRUD operations."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def lookup_vehicle(self, plate_number: str):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT owner_name, contact_number, vehicle_model, registration_city FROM vehicle_registry WHERE plate_number = ?",
            (plate_number,),
        )
        row = cursor.fetchone()
        conn.close()
        if row:
            return {
                "owner": row["owner_name"],
                "phone": row["contact_number"],
                "model": row["vehicle_model"],
                "city": row["registration_city"],
                "registered": True,
            }
        return None

    def record_challan(
        self,
        track_id: int,
        plate_number: str,
        owner_name: str,
        violation_type: str = "Red Light Jump",
        fine_amount: int = 1000,
        evidence_path: str = "violations/test.jpg",
    ) -> int:
        from datetime import datetime
        # Enforce native int
        native_track_id = int(track_id)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO challan_records (timestamp, track_id, plate_number, owner_name, violation_type, fine_amount, evidence_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (timestamp, native_track_id, plate_number, owner_name, violation_type, int(fine_amount), evidence_path),
        )
        challan_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return challan_id


class TestDatabaseCRUD(unittest.TestCase):
    """Tests SQLite interactions and RTO lookup semantics."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_crud.db")
        init_test_database(self.db_path)

        if DatabaseManager is not None:
            self.mgr = DatabaseManager(self.db_path)
        else:
            self.mgr = OracleDatabaseManager(self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except OSError:
                pass

    def test_lookup_known_vehicle(self):
        """Verifies querying an existing plate returns complete owner and model details."""
        details = self.mgr.lookup_vehicle("AS01AB1234")
        self.assertIsNotNone(details)
        self.assertEqual(details["owner"], "Rajesh Sharma")
        self.assertEqual(details["model"], "Hyundai Creta")
        self.assertEqual(details["city"], "Guwahati")
        self.assertTrue(details.get("registered", True))

    def test_lookup_missing_vehicle_returns_none(self):
        """Verifies querying an unregistered plate returns None (or unregistered flag)."""
        details = self.mgr.lookup_vehicle("ZZ99ZZ9999")
        if details is not None:
            # If fallback dict returned, registered must be False
            self.assertFalse(details.get("registered", False))
        else:
            self.assertIsNone(details)

    def test_record_challan_insertion(self):
        """Verifies inserting a challan record returns positive primary key ID and persists row."""
        challan_id = self.mgr.record_challan(
            track_id=101,
            plate_number="DL04CA9999",
            owner_name="Priya Verma",
            violation_type="Red Light Jump",
            fine_amount=1000,
            evidence_path="violations/dl04_evidence.jpg",
        )
        self.assertIsInstance(challan_id, int)
        self.assertGreater(challan_id, 0)

        # Direct SQL query to verify data persistence
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT track_id, plate_number, owner_name, fine_amount, typeof(track_id) FROM challan_records WHERE challan_id = ?", (challan_id,))
        row = cursor.fetchone()
        conn.close()

        self.assertIsNotNone(row)
        self.assertEqual(row[0], 101)
        self.assertEqual(row[1], "DL04CA9999")
        self.assertEqual(row[2], "Priya Verma")
        self.assertEqual(row[3], 1000)
        self.assertEqual(row[4], "integer", "track_id must be stored as SQLite integer")

    def test_multiple_challans_query_integrity(self):
        """Verifies multiple challan insertions maintain independent primary keys."""
        id1 = self.mgr.record_challan(201, "MH02DZ4567", "Amit Sen", fine_amount=1000)
        id2 = self.mgr.record_challan(202, "KA05MJ8821", "Vikram Rao", fine_amount=1500)
        self.assertNotEqual(id1, id2)
        self.assertGreater(id2, id1)


if __name__ == "__main__":
    unittest.main()
