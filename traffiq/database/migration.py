"""
Migration and data sanitization routines for SQLite challan records.
Remediates binary blob serialization defects into native integer values.
"""

from __future__ import annotations

import re
import sqlite3
import struct
from typing import Optional, Tuple

from traffiq.database.connection import get_connection, register_sqlite_adapters


def decode_track_id_blob(blob_val: bytes | memoryview | bytearray, evidence_path: str = "") -> int:
    """
    Decodes binary buffer of track_id into native Python int.
    Handles 8-byte / 4-byte little-endian integers, with fallback to evidence_path extraction.
    """
    buffer = bytes(blob_val) if isinstance(blob_val, (memoryview, bytearray)) else blob_val
    recovered_id: Optional[int] = None

    # Attempt binary decoding based on buffer length
    if isinstance(buffer, bytes) and len(buffer) > 0:
        try:
            if len(buffer) == 8:
                recovered_id = int.from_bytes(buffer, byteorder="little", signed=True)
            elif len(buffer) == 4:
                recovered_id = int.from_bytes(buffer, byteorder="little", signed=True)
            elif len(buffer) == 2:
                recovered_id = int.from_bytes(buffer, byteorder="little", signed=True)
            elif len(buffer) == 1:
                recovered_id = int.from_bytes(buffer, byteorder="little", signed=True)
            elif len(buffer) >= 8:
                recovered_id = int(struct.unpack("<q", buffer[:8])[0])
        except Exception:
            recovered_id = None

    # Fallback to evidence path extraction if buffer could not be parsed or produced non-positive ID
    if recovered_id is None or recovered_id <= 0:
        match = re.search(r"violation_ID(\d+)_", str(evidence_path))
        if match:
            recovered_id = int(match.group(1))

    if recovered_id is None:
        raise ValueError(
            f"Unable to recover track_id from blob buffer ({buffer!r}) and evidence_path ({evidence_path!r})"
        )

    return int(recovered_id)


def migrate_legacy_blobs(db_path: str = "traffiq_enforcement.db") -> int:
    """
    In-place migration function that finds all rows in challan_records where
    typeof(track_id) = 'blob', decodes binary buffer, casts to native Python int,
    and updates the row in-place.

    Returns the count of successfully migrated rows.
    """
    register_sqlite_adapters()
    conn = get_connection(db_path)
    migrated_count = 0

    try:
        cursor = conn.cursor()

        # Query all rows where track_id is stored as BLOB
        cursor.execute(
            """
            SELECT challan_id, track_id, evidence_path
            FROM challan_records
            WHERE typeof(track_id) = 'blob'
            """
        )
        blob_rows = cursor.fetchall()

        for row in blob_rows:
            challan_id = row["challan_id"]
            blob_val = row["track_id"]
            evidence_path = row["evidence_path"]

            clean_track_id = decode_track_id_blob(blob_val, evidence_path=evidence_path)

            cursor.execute(
                """
                UPDATE challan_records
                SET track_id = ?
                WHERE challan_id = ?
                """,
                (int(clean_track_id), challan_id),
            )
            migrated_count += 1

        conn.commit()
    finally:
        conn.close()

    return migrated_count


def verify_integrity(db_path: str = "traffiq_enforcement.db") -> Tuple[int, int, int]:
    """
    Inspects challan_records table and returns:
    (total_rows, integer_rows, blob_rows).
    """
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM challan_records")
        total_rows = int(cursor.fetchone()[0])

        cursor.execute(
            "SELECT COUNT(*) FROM challan_records WHERE typeof(track_id) = 'integer'"
        )
        integer_rows = int(cursor.fetchone()[0])

        cursor.execute(
            "SELECT COUNT(*) FROM challan_records WHERE typeof(track_id) = 'blob'"
        )
        blob_rows = int(cursor.fetchone()[0])

        return (total_rows, integer_rows, blob_rows)
    finally:
        conn.close()


# Alias for test suite compatibility
verify_database_integrity = verify_integrity
