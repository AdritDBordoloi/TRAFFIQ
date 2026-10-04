"""
Repository and DatabaseManager for TRAFFIQ e-challan and vehicle registry operations.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional, Tuple

from traffiq.database.connection import create_connection, register_sqlite_adapters
from traffiq.database.migration import migrate_legacy_blobs, verify_integrity
from traffiq.database.schema import init_schema


class DatabaseManager:
    """
    Central database manager handling vehicle registry queries,
    e-challan generation with strict integer enforcement, and integrity verification.
    """

    def __init__(self, db_path: str = "traffiq_enforcement.db") -> None:
        self.db_path = db_path
        register_sqlite_adapters()
        self._memory_conn: Optional[sqlite3.Connection] = None
        if self.db_path == ":memory:" or "mode=memory" in self.db_path:
            self._memory_conn = create_connection(self.db_path)
        self._ensure_initialized()

    def _ensure_initialized(self) -> None:
        """Ensure schema exists and initial seed data is present if table is empty."""
        with self._connection_scope() as conn:
            init_schema(conn)

    def get_connection(self) -> sqlite3.Connection:
        """Create and return a configured SQLite connection."""
        if self._memory_conn is not None:
            return self._memory_conn
        return create_connection(self.db_path)

    @contextmanager
    def _connection_scope(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield an active connection, closing it on exit unless it is persistent in-memory."""
        conn = self.get_connection()
        try:
            yield conn
        finally:
            if self._memory_conn is None:
                conn.close()

    def close(self) -> None:
        """Close any persistent in-memory database connection."""
        if self._memory_conn is not None:
            try:
                self._memory_conn.close()
            except Exception:
                pass
            self._memory_conn = None

    def lookup_vehicle(self, plate_number: str) -> Optional[Dict[str, Any]]:
        """
        Look up a vehicle by license plate number.
        Returns a dictionary with owner details or None if not found.
        """
        if not plate_number:
            return None

        normalized_plate = plate_number.strip().upper()
        with self._connection_scope() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT plate_number, owner_name, contact_number, vehicle_model, registration_city
                FROM vehicle_registry
                WHERE UPPER(plate_number) = ?
                """,
                (normalized_plate,),
            )
            row = cursor.fetchone()
            if row is None:
                return None

            return {
                "plate_number": row["plate_number"],
                "owner_name": row["owner_name"],
                "owner": row["owner_name"],  # Alias for compatibility
                "contact_number": row["contact_number"],
                "contact": row["contact_number"],  # Alias
                "phone": row["contact_number"],  # Alias
                "vehicle_model": row["vehicle_model"],
                "model": row["vehicle_model"],  # Alias
                "registration_city": row["registration_city"],
                "city": row["registration_city"],  # Alias
                "registered": True,
            }

    def add_vehicle(
        self,
        plate_number: str,
        owner_name: str,
        contact_number: str,
        vehicle_model: str,
        registration_city: str,
    ) -> None:
        """Register or update a vehicle in vehicle_registry."""
        with self._connection_scope() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO vehicle_registry (
                    plate_number, owner_name, contact_number, vehicle_model, registration_city
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    plate_number.strip().upper(),
                    owner_name.strip(),
                    contact_number.strip(),
                    vehicle_model.strip(),
                    registration_city.strip(),
                ),
            )
            conn.commit()

    def record_challan(
        self,
        track_id: int,
        plate_number: str,
        owner_name: str,
        violation_type: str = "Red Light Jump",
        fine_amount: int = 1000,
        evidence_path: str = "",
        timestamp: Optional[str] = None,
    ) -> int:
        """
        Record a traffic violation challan into challan_records.
        Strictly coerces and validates that track_id is bound as a native Python int.
        Returns the inserted challan_id.
        """
        # Strict integer coercion & validation
        try:
            clean_track_id = int(track_id)
        except (TypeError, ValueError) as err:
            raise TypeError(
                f"track_id must be coercible to int, got {type(track_id)}: {track_id}"
            ) from err

        try:
            clean_fine = int(fine_amount)
        except (TypeError, ValueError) as err:
            raise TypeError(
                f"fine_amount must be coercible to int, got {type(fine_amount)}: {fine_amount}"
            ) from err

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        clean_plate = str(plate_number).strip().upper()
        clean_owner = str(owner_name).strip()
        clean_violation = str(violation_type).strip()
        clean_evidence = str(evidence_path).strip()

        with self._connection_scope() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO challan_records (
                    timestamp, track_id, plate_number, owner_name, violation_type, fine_amount, evidence_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    clean_track_id,
                    clean_plate,
                    clean_owner,
                    clean_violation,
                    clean_fine,
                    clean_evidence,
                ),
            )
            conn.commit()
            challan_id = int(cursor.lastrowid)
            return challan_id

    def get_all_challans(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve all issued challan records."""
        with self._connection_scope() as conn:
            cursor = conn.cursor()
            if limit is not None:
                cursor.execute(
                    "SELECT * FROM challan_records ORDER BY challan_id DESC LIMIT ?",
                    (int(limit),),
                )
            else:
                cursor.execute("SELECT * FROM challan_records ORDER BY challan_id DESC")

            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_challan_count(self) -> int:
        """Return the total number of issued challans."""
        with self._connection_scope() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM challan_records")
            return int(cursor.fetchone()[0])

    def get_challan_by_id(self, challan_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a single challan by primary key ID."""
        with self._connection_scope() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM challan_records WHERE challan_id = ?",
                (int(challan_id),),
            )
            row = cursor.fetchone()
            return dict(row) if row is not None else None

    def get_challans_by_plate(self, plate_number: str) -> List[Dict[str, Any]]:
        """Retrieve all challans for a given license plate."""
        with self._connection_scope() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM challan_records WHERE UPPER(plate_number) = ? ORDER BY challan_id DESC",
                (plate_number.strip().upper(),),
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def migrate_legacy_blobs(self) -> int:
        """Migrate any legacy BLOB track_id entries in-place to native integers."""
        return migrate_legacy_blobs(self.db_path)

    def verify_integrity(self) -> Tuple[int, int, int]:
        """Verify database integrity; returns (total_rows, integer_rows, blob_rows)."""
        return verify_integrity(self.db_path)


def lookup_vehicle(
    plate_number: str, db_path: str = "traffiq_enforcement.db"
) -> Optional[Dict[str, Any]]:
    """Module-level convenience function to look up vehicle details."""
    mgr = DatabaseManager(db_path)
    return mgr.lookup_vehicle(plate_number)


def record_challan(
    track_id: int,
    plate_number: str,
    owner_name: str,
    violation_type: str = "Red Light Jump",
    fine_amount: int = 1000,
    evidence_path: str = "",
    timestamp: Optional[str] = None,
    db_path: str = "traffiq_enforcement.db",
) -> int:
    """Module-level convenience function to record challan with strict int track_id."""
    mgr = DatabaseManager(db_path)
    return mgr.record_challan(
        track_id=track_id,
        plate_number=plate_number,
        owner_name=owner_name,
        violation_type=violation_type,
        fine_amount=fine_amount,
        evidence_path=evidence_path,
        timestamp=timestamp,
    )
