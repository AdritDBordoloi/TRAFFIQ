"""TRAFFIQ Database Setup Utility (Consolidated)."""
from traffiq.database.schema import setup_database

def init_db(db_path="traffiq_enforcement.db"):
    """Initialize database tables and seed RTO records."""
    setup_database(db_path)
    print(f"[SUCCESS] {db_path} ready with RTO vehicle records.")

if __name__ == "__main__":
    init_db()
