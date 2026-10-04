"""Shared test fixtures, synthetic frame generators, and isolated database utilities
for the TRAFFIQ automated test suite.
"""

import os
import sys
import sqlite3
import tempfile
import numpy as np

# Ensure project root is present in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Try importing pytest for fixture decoration; if not installed, provide no-op decorator
try:
    import pytest
except ImportError:
    class DummyPytest:
        @staticmethod
        def fixture(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
    pytest = DummyPytest()


# ---------------------------------------------------------------------------
# Synthetic Frame & Crop Generators
# ---------------------------------------------------------------------------

def generate_synthetic_frame(width: int = 1280, height: int = 720) -> np.ndarray:
    """Generates a synthetic 3-channel BGR roadway frame.
    Includes simulated asphalt background, stop line, and lane markings.
    """
    frame = np.full((height, width, 3), 40, dtype=np.uint8)  # Dark asphalt gray
    
    # Yellow center line
    frame[:, width // 2 - 2 : width // 2 + 2] = [0, 255, 255]
    
    # White stop line at 55% height
    stop_y = int(height * 0.55)
    frame[stop_y - 2 : stop_y + 2, :] = [255, 255, 255]
    
    return frame


def generate_synthetic_vehicle_crop(
    width: int = 300, height: int = 200, has_plate: bool = True
) -> np.ndarray:
    """Generates a synthetic vehicle crop with a simulated license plate region
    located in the lower 40-50% horizontal band.
    """
    crop = np.full((height, width, 3), 120, dtype=np.uint8)  # Vehicle body gray
    
    if has_plate:
        # Plate region in the lower 40-50% band (y in [height*0.6, height*0.85], x centered)
        plate_y1 = int(height * 0.65)
        plate_y2 = int(height * 0.85)
        plate_x1 = int(width * 0.25)
        plate_x2 = int(width * 0.75)
        
        # White background plate with black border
        crop[plate_y1:plate_y2, plate_x1:plate_x2] = [250, 250, 250]
        crop[plate_y1 : plate_y1 + 2, plate_x1:plate_x2] = [0, 0, 0]
        crop[plate_y2 - 2 : plate_y2, plate_x1:plate_x2] = [0, 0, 0]
        crop[plate_y1:plate_y2, plate_x1 : plate_x1 + 2] = [0, 0, 0]
        crop[plate_y1:plate_y2, plate_x2 - 2 : plate_x2] = [0, 0, 0]
    
    return crop


def generate_synthetic_blank_crop(
    width: int = 200, height: int = 100, color: int = 0
) -> np.ndarray:
    """Generates a uniform color image (default pure black, or white if color=255)."""
    return np.full((height, width, 3), color, dtype=np.uint8)


def generate_synthetic_noisy_crop(width: int = 200, height: int = 100) -> np.ndarray:
    """Generates an image filled with uniform random noise."""
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Database Utilities & Schemas
# ---------------------------------------------------------------------------

SAMPLE_RTO_VEHICLES = [
    ("AS01AB1234", "Rajesh Sharma", "+91 98765 43210", "Hyundai Creta", "Guwahati"),
    ("DL04CA9999", "Priya Verma", "+91 98111 22334", "Maruti Swift", "Delhi"),
    ("MH02DZ4567", "Amit Sen", "+91 97234 56789", "Honda City", "Mumbai"),
    ("KA05MJ8821", "Vikram Rao", "+91 99001 12233", "Toyota Innova", "Bengaluru"),
    ("SAMPLE123", "Demo Driver", "+91 90000 00000", "Test Vehicle", "Local Area"),
]


def init_test_database(db_path: str) -> None:
    """Initializes a SQLite database with vehicle_registry and challan_records schemas."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS vehicle_registry (
        plate_number TEXT PRIMARY KEY,
        owner_name TEXT NOT NULL,
        contact_number TEXT NOT NULL,
        vehicle_model TEXT NOT NULL,
        registration_city TEXT NOT NULL
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS challan_records (
        challan_id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        track_id INT NOT NULL,
        plate_number TEXT NOT NULL,
        owner_name TEXT NOT NULL,
        violation_type TEXT NOT NULL,
        fine_amount INT NOT NULL,
        evidence_path TEXT NOT NULL
    )
    """)
    cursor.executemany("""
    INSERT OR REPLACE INTO vehicle_registry (plate_number, owner_name, contact_number, vehicle_model, registration_city)
    VALUES (?, ?, ?, ?, ?)
    """, SAMPLE_RTO_VEHICLES)
    conn.commit()
    conn.close()


def populate_corrupted_blob_records(db_path: str, count: int = 5) -> None:
    """Populates legacy challan records containing raw binary blob track_id entries."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    for i in range(1, count + 1):
        # Convert int to 8-byte little-endian binary blob
        track_id_blob = int(100 + i).to_bytes(8, byteorder="little", signed=True)
        cursor.execute("""
        INSERT INTO challan_records (timestamp, track_id, plate_number, owner_name, violation_type, fine_amount, evidence_path)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            f"2026-10-04 12:00:0{i}",
            sqlite3.Binary(track_id_blob),
            f"DL04CA990{i}",
            f"Legacy Owner {i}",
            "Red Light Jump",
            1000,
            f"violations/legacy_{i}.jpg",
        ))
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Pytest Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_frame():
    return generate_synthetic_frame()


@pytest.fixture
def synthetic_vehicle_crop():
    return generate_synthetic_vehicle_crop()


@pytest.fixture
def isolated_db_path(tmp_path=None):
    """Provides path to an isolated, temporary SQLite database with valid schemas."""
    if tmp_path is not None:
        db_file = os.path.join(str(tmp_path), "test_traffiq.db")
    else:
        temp_dir = tempfile.mkdtemp()
        db_file = os.path.join(temp_dir, "test_traffiq.db")
    init_test_database(db_file)
    return db_file


@pytest.fixture
def corrupted_db_path(tmp_path=None):
    """Provides path to a temporary SQLite database with corrupted blob track_ids."""
    if tmp_path is not None:
        db_file = os.path.join(str(tmp_path), "corrupted_traffiq.db")
    else:
        temp_dir = tempfile.mkdtemp()
        db_file = os.path.join(temp_dir, "corrupted_traffiq.db")
    init_test_database(db_file)
    populate_corrupted_blob_records(db_file, count=5)
    return db_file
