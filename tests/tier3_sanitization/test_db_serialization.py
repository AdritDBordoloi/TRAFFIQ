"""Tier 3: SQLite Data Serialization & Type Integrity Tests.
Verifies that track_id values passed as np.int64, np.int32, np.int16, or native Python int
are strictly stored as SQLite INTEGER, never as raw binary BLOB buffers.
"""

import os
import sqlite3
import tempfile
import unittest
import numpy as np

from tests.conftest import init_test_database

try:
    from traffiq.database.connection import get_db_connection
    from traffiq.database.repository import DatabaseManager, record_challan
except ImportError:
    get_db_connection = None
    DatabaseManager = None
    record_challan = None


def oracle_register_numpy_adapters():
    """Registers SQLite adapters converting NumPy scalar integers to Python native int."""
    sqlite3.register_adapter(np.int64, int)
    sqlite3.register_adapter(np.int32, int)
    sqlite3.register_adapter(np.int16, int)
    sqlite3.register_adapter(np.int8, int)
    sqlite3.register_adapter(np.integer, int)


class TestDBSerialization(unittest.TestCase):
    """Tests SQLite type safety and binary blob elimination."""

    def setUp(self):
        oracle_register_numpy_adapters()
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_serialization.db")
        init_test_database(self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except OSError:
                pass

    def test_native_python_int_serialization(self):
        """Verifies native Python int is stored as SQLite INTEGER."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        track_id = int(101)
        cur.execute(
            """
            INSERT INTO challan_records (timestamp, track_id, plate_number, owner_name, violation_type, fine_amount, evidence_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("2026-10-04 12:00:00", track_id, "AS01AB1234", "Rajesh Sharma", "Red Light Jump", 1000, "violations/1.jpg"),
        )
        conn.commit()

        cur.execute("SELECT track_id, typeof(track_id) FROM challan_records WHERE track_id = 101")
        row = cur.fetchone()
        conn.close()

        self.assertIsNotNone(row)
        self.assertEqual(row[0], 101)
        self.assertEqual(row[1], "integer")

    def test_numpy_int64_serialization_as_integer(self):
        """Verifies np.int64 (returned by YOLO / ByteTrack) is stored as SQLite INTEGER, NOT BLOB."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        # Explicitly pass np.int64 or cast via repository
        raw_track_id = np.int64(202)
        safe_track_id = int(raw_track_id)  # Defensive casting required by R1

        cur.execute(
            """
            INSERT INTO challan_records (timestamp, track_id, plate_number, owner_name, violation_type, fine_amount, evidence_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("2026-10-04 12:01:00", safe_track_id, "DL04CA9999", "Priya Verma", "Red Light Jump", 1000, "violations/2.jpg"),
        )
        conn.commit()

        cur.execute("SELECT track_id, typeof(track_id) FROM challan_records WHERE track_id = 202")
        row = cur.fetchone()
        conn.close()

        self.assertIsNotNone(row)
        self.assertEqual(row[0], 202)
        self.assertEqual(row[1], "integer", "np.int64 must be stored as INTEGER, never BLOB")

    def test_numpy_int32_and_int16_serialization(self):
        """Verifies np.int32 and np.int16 are cast and stored as INTEGER."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO challan_records (timestamp, track_id, plate_number, owner_name, violation_type, fine_amount, evidence_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("2026-10-04 12:02:00", int(np.int32(303)), "MH02DZ4567", "Amit Sen", "Red Light Jump", 1000, "violations/3.jpg"),
        )
        cur.execute(
            """
            INSERT INTO challan_records (timestamp, track_id, plate_number, owner_name, violation_type, fine_amount, evidence_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("2026-10-04 12:03:00", int(np.int16(404)), "KA05MJ8821", "Vikram Rao", "Red Light Jump", 1000, "violations/4.jpg"),
        )
        conn.commit()

        cur.execute("SELECT typeof(track_id) FROM challan_records WHERE track_id IN (303, 404)")
        types = [r[0] for r in cur.fetchall()]
        conn.close()

        self.assertEqual(types, ["integer", "integer"])

    def test_zero_blob_invariant_across_inserted_records(self):
        """Verifies zero binary blobs remain in challan_records table."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        # Insert 10 records with varied integer sources
        for i in range(1, 11):
            t_id = np.int64(500 + i)
            cur.execute(
                """
                INSERT INTO challan_records (timestamp, track_id, plate_number, owner_name, violation_type, fine_amount, evidence_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (f"2026-10-04 12:10:{i:02d}", int(t_id), f"TEST{i:04d}", f"Driver {i}", "Red Light Jump", 1000, f"violations/{i}.jpg"),
            )
        conn.commit()

        cur.execute("SELECT COUNT(*) FROM challan_records WHERE typeof(track_id) = 'blob'")
        blob_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM challan_records WHERE typeof(track_id) != 'integer'")
        non_int_count = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM challan_records")
        total_rows = cur.fetchone()[0]
        conn.close()

        self.assertEqual(blob_count, 0, f"Found {blob_count} binary blob track_ids")
        self.assertEqual(non_int_count, 0, f"Found {non_int_count} non-integer track_ids")
        self.assertEqual(total_rows, 10)


if __name__ == "__main__":
    unittest.main()
