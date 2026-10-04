"""
Database submodule for TRAFFIQ: RTO vehicle registry, challan records,
NumPy SQLite adapters, thread-safe connections, and legacy blob migration.
"""

from traffiq.database.connection import (
    create_connection,
    get_connection,
    get_db_connection,
    register_sqlite_adapters,
)
from traffiq.database.migration import (
    decode_track_id_blob,
    migrate_legacy_blobs,
    verify_database_integrity,
    verify_integrity,
)
from traffiq.database.repository import (
    DatabaseManager,
    lookup_vehicle,
    record_challan,
)
from traffiq.database.schema import (
    SAMPLE_VEHICLES,
    init_schema,
    seed_sample_vehicles,
    setup_database,
)

__all__ = [
    "DatabaseManager",
    "create_connection",
    "get_connection",
    "get_db_connection",
    "register_sqlite_adapters",
    "lookup_vehicle",
    "record_challan",
    "decode_track_id_blob",
    "migrate_legacy_blobs",
    "verify_integrity",
    "verify_database_integrity",
    "init_schema",
    "seed_sample_vehicles",
    "setup_database",
    "SAMPLE_VEHICLES",
]
