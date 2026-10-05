import os
import sqlite3

def display_all_challans(db_path="traffiq_enforcement.db"):
    if not os.path.exists(db_path):
        print("\n" + "=" * 90)
        print("                    TRAFFIQ E-CHALLAN AUDIT LOG (DATABASE)                   ")
        print("=" * 90)
        print(f"Database file '{db_path}' not found. Run the enforcement pipeline first.")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='challan_records'")
    if not cursor.fetchone():
        print("\n" + "=" * 90)
        print("                    TRAFFIQ E-CHALLAN AUDIT LOG (DATABASE)                   ")
        print("=" * 90)
        print("No 'challan_records' table found. Run init_db.py or the enforcement pipeline first.")
        conn.close()
        return

    cursor.execute("""
        SELECT challan_id, timestamp, track_id, plate_number, owner_name, contact_number, vehicle_model, fine_amount, full_evidence_path
        FROM challan_records
        ORDER BY challan_id DESC
    """)
    rows = cursor.fetchall()
    conn.close()

    print("\n" + "=" * 90)
    print("                    TRAFFIQ E-CHALLAN AUDIT LOG (DATABASE)                   ")
    print("=" * 90)
    if not rows:
        print("No violations recorded yet. Run live_enforcement.py to generate tickets.")
        return

    for r in rows:
        print(f"Challan ID : TRFQ-{r[0]:06d} | Time: {r[1]} | Track: #{r[2]}")
        print(f"Plate      : {r[3]:<12} | Owner: {r[4]} ({r[5]})")
        print(f"Vehicle    : {r[6]:<15} | Penalty: INR {r[7]}")
        print(f"Evidence   : {r[8]}")
        print("-" * 90)

if __name__ == "__main__":
    display_all_challans()