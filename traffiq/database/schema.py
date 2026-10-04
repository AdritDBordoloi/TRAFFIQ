"""
Database schema definitions, DDL execution, and sample seeding for TRAFFIQ.
"""

from __future__ import annotations

import sqlite3
from typing import List, Optional, Sequence, Tuple

# DDL Statements
CREATE_VEHICLE_REGISTRY_SQL = """
CREATE TABLE IF NOT EXISTS vehicle_registry (
    plate_number TEXT PRIMARY KEY,
    owner_name TEXT NOT NULL,
    contact_number TEXT NOT NULL,
    vehicle_model TEXT NOT NULL,
    registration_city TEXT NOT NULL
);
"""

CREATE_CHALLAN_RECORDS_SQL = """
CREATE TABLE IF NOT EXISTS challan_records (
    challan_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    track_id INT NOT NULL,
    plate_number TEXT NOT NULL,
    owner_name TEXT NOT NULL,
    violation_type TEXT NOT NULL,
    fine_amount INT NOT NULL,
    evidence_path TEXT NOT NULL
);
"""

CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_challan_track_id ON challan_records (track_id);",
    "CREATE INDEX IF NOT EXISTS idx_challan_plate ON challan_records (plate_number);",
    "CREATE INDEX IF NOT EXISTS idx_challan_timestamp ON challan_records (timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_vehicle_owner ON vehicle_registry (owner_name);",
]

# Standard Exhibition Seed Records (from init_db.py)
SAMPLE_VEHICLES: List[Tuple[str, str, str, str, str]] = [
    ("AS01AB1234", "Rajesh Sharma", "+91 98765 43210", "Hyundai Creta", "Guwahati"),
    ("DL04CA9999", "Priya Verma", "+91 98111 22334", "Maruti Swift", "Delhi"),
    ("MH02DZ4567", "Amit Sen", "+91 97234 56789", "Honda City", "Mumbai"),
    ("KA05MJ8821", "Vikram Rao", "+91 99001 12233", "Toyota Innova", "Bengaluru"),
    ("SAMPLE123", "Demo Driver", "+91 90000 00000", "Test Vehicle", "Local Area"),
]


def seed_sample_vehicles(
    conn: sqlite3.Connection,
    vehicles: Optional[Sequence[Tuple[str, str, str, str, str]]] = None,
) -> int:
    """
    Seed sample vehicle registrations if vehicle_registry is empty.
    Returns the count of inserted records.
    """
    records_to_seed = vehicles if vehicles is not None else SAMPLE_VEHICLES
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM vehicle_registry")
    count = cursor.fetchone()[0]

    if count == 0:
        cursor.executemany(
            """
            INSERT OR REPLACE INTO vehicle_registry (
                plate_number, owner_name, contact_number, vehicle_model, registration_city
            ) VALUES (?, ?, ?, ?, ?)
            """,
            records_to_seed,
        )
        conn.commit()
        return len(records_to_seed)
    return 0


def init_schema(conn: sqlite3.Connection) -> None:
    """
    Execute DDL to create tables, indices, and seed sample vehicle data if empty.
    """
    cursor = conn.cursor()
    cursor.execute(CREATE_VEHICLE_REGISTRY_SQL)
    cursor.execute(CREATE_CHALLAN_RECORDS_SQL)

    for index_stmt in CREATE_INDEXES_SQL:
        cursor.execute(index_stmt)

    conn.commit()
    seed_sample_vehicles(conn)


def setup_database(db_path: str = "traffiq_enforcement.db") -> None:
    """
    Initializes database schema and seed data for the specified SQLite file.
    """
    conn = sqlite3.connect(db_path)
    try:
        init_schema(conn)
    finally:
        conn.close()
