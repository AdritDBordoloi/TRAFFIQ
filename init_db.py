# init_db.py
import sqlite3

def setup_database():
    conn = sqlite3.connect("traffiq_enforcement.db")
    cursor = conn.cursor()

    # 1. RTO Vehicle Registry Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS vehicle_registry (
        plate_number TEXT PRIMARY KEY,
        owner_name TEXT NOT NULL,
        contact_number TEXT NOT NULL,
        vehicle_model TEXT NOT NULL,
        registration_city TEXT NOT NULL
    )
    """)

    # 2. Issued E-Challans Table
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

    # Populate with sample vehicle registrations for exhibition testing
    sample_vehicles = [
        ("AS01AB1234", "Rajesh Sharma", "+91 98765 43210", "Hyundai Creta", "Guwahati"),
        ("DL04CA9999", "Priya Verma", "+91 98111 22334", "Maruti Swift", "Delhi"),
        ("MH02DZ4567", "Amit Sen", "+91 97234 56789", "Honda City", "Mumbai"),
        ("KA05MJ8821", "Vikram Rao", "+91 99001 12233", "Toyota Innova", "Bengaluru"),
        ("SAMPLE123",  "Demo Driver", "+91 90000 00000", "Test Vehicle", "Local Area")
    ]

    cursor.executemany("""
    INSERT OR REPLACE INTO vehicle_registry (plate_number, owner_name, contact_number, vehicle_model, registration_city)
    VALUES (?, ?, ?, ?, ?)
    """, sample_vehicles)

    conn.commit()
    conn.close()
    print("[SUCCESS] traffiq_enforcement.db initialized with sample RTO records.")

if __name__ == "__main__":
    setup_database()