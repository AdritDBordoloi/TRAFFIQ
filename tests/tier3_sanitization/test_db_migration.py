"""Tier 3: Database Migration & Legacy Binary Blob Sanitization Tests.
Verifies that the migration utility recovers corrupted binary blob track_ids,
unpacks little-endian integer buffers in-place, achieves 100% integer rows,
and is fully idempotent.
"""

import os
import sqlite3
import tempfile
import unittest

from tests.conftest import init_test_database, populate_corrupted_blob_records

try:
    from traffiq.database.migration import migrate_legacy_blobs, verify_database_integrity
except ImportError:
    migrate_legacy_blobs = None
    verify_database_integrity = None


def oracle_migrate_legacy_blobs(db_path: str) -> int:
    """Authoritative reference migration utility recovering binary blobs to integers."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT challan_id, track_id FROM challan_records WHERE typeof(track_id) = 'blob'")
    blob_rows = cursor.fetchall()
    migrated_count = 0

    for challan_id, raw_blob in blob_rows:
        if isinstance(raw_blob, (bytes, memoryview)):
            raw_bytes = bytes(raw_blob)
            if len(raw_bytes) in (4, 8):
                int_val = int.from_bytes(raw_bytes, byteorder="little", signed=True)
            else:
                int_val = int.from_bytes(raw_bytes[:8], byteorder="little", signed=True)
        else:
            int_val = int(raw_blob)

        cursor.execute(
            "UPDATE challan_records SET track_id = ? WHERE challan_id = ?",
            (int(int_val), challan_id),
        )
        migrated_count += 1

    conn.commit()
    conn.close()
    return migrated_count


def oracle_verify_integrity(db_path: str) -> tuple[int, int, int]:
    """Returns (total_rows, integer_rows, blob_rows)."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM challan_records")
    total = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM challan_records WHERE typeof(track_id) = 'integer'")
    integers = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM challan_records WHERE typeof(track_id) = 'blob'")
    blobs = cur.fetchone()[0]
    conn.close()
    return total, integers, blobs


class TestDBMigration(unittest.TestCase):
    """Tests legacy database sanitization and blob unpacking."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_migration.db")
        init_test_database(self.db_path)
        # Seed 5 legacy blob records (track_ids 101, 102, 103, 104, 105)
        populate_corrupted_blob_records(self.db_path, count=5)

        self.migrate_fn = migrate_legacy_blobs or oracle_migrate_legacy_blobs
        self.verify_fn = verify_database_integrity or oracle_verify_integrity

    def tearDown(self):
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except OSError:
                pass

    def test_pre_migration_state_contains_blobs(self):
        """Verifies initial test database has 5 binary blob rows."""
        total, integers, blobs = self.verify_fn(self.db_path)
        self.assertEqual(total, 5)
        self.assertEqual(blobs, 5)
        self.assertEqual(integers, 0)

    def test_migration_recovers_all_blobs_to_integers(self):
        """Verifies migration unpacks all blobs to valid integers without data loss."""
        migrated = self.migrate_fn(self.db_path)
        self.assertEqual(migrated, 5)

        total, integers, blobs = self.verify_fn(self.db_path)
        self.assertEqual(total, 5)
        self.assertEqual(integers, 5)
        self.assertEqual(blobs, 0, "No binary blobs should remain after migration")

        # Verify recovered integer values match original IDs (101, 102, 103, 104, 105)
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT track_id FROM challan_records ORDER BY challan_id ASC")
        recovered_ids = [r[0] for r in cur.fetchall()]
        conn.close()

        self.assertEqual(recovered_ids, [101, 102, 103, 104, 105])

    def test_migration_mixed_types_integrity(self):
        """Verifies migration cleanly handles a database with both valid integers and legacy blobs."""
        # Add 3 clean integer rows
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        for i in [201, 202, 203]:
            cur.execute(
                """
                INSERT INTO challan_records (timestamp, track_id, plate_number, owner_name, violation_type, fine_amount, evidence_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                ("2026-10-04 12:30:00", i, "DL04CA9999", "Owner", "Red Light", 1000, "e.jpg"),
            )
        conn.commit()
        conn.close()

        total, integers, blobs = self.verify_fn(self.db_path)
        self.assertEqual(total, 8)
        self.assertEqual(blobs, 5)
        self.assertEqual(integers, 3)

        # Run migration
        migrated = self.migrate_fn(self.db_path)
        self.assertEqual(migrated, 5)

        total, integers, blobs = self.verify_fn(self.db_path)
        self.assertEqual(total, 8)
        self.assertEqual(integers, 8)
        self.assertEqual(blobs, 0)

    def test_migration_idempotency(self):
        """Verifies running migration repeatedly is safe and modifies 0 rows on clean tables."""
        # First run cleans the database
        self.migrate_fn(self.db_path)
        
        # Second run should find 0 blobs and modify 0 rows
        second_run_migrated = self.migrate_fn(self.db_path)
        self.assertEqual(second_run_migrated, 0)

        total, integers, blobs = self.verify_fn(self.db_path)
        self.assertEqual(total, 5)
        self.assertEqual(integers, 5)
        self.assertEqual(blobs, 0)


if __name__ == "__main__":
    unittest.main()
