"""
Thread-safe SQLite connection factory and NumPy adapter registration for TRAFFIQ.
"""

from __future__ import annotations

import sqlite3
import threading

_ADAPTERS_REGISTERED = False
_LOCK = threading.Lock()


def register_sqlite_adapters() -> None:
    """
    Register SQLite adapters to serialize NumPy scalars to native Python types.
    This prevents numpy.int64 / numpy.int32 from serializing as binary blobs (BLOBs).
    """
    global _ADAPTERS_REGISTERED
    with _LOCK:
        if _ADAPTERS_REGISTERED:
            return

        try:
            import numpy as np

            # Register integer adapters
            sqlite3.register_adapter(np.integer, lambda val: int(val))
            sqlite3.register_adapter(np.int64, lambda val: int(val))
            sqlite3.register_adapter(np.int32, lambda val: int(val))
            sqlite3.register_adapter(np.int16, lambda val: int(val))
            sqlite3.register_adapter(np.int8, lambda val: int(val))
            sqlite3.register_adapter(np.uint64, lambda val: int(val))
            sqlite3.register_adapter(np.uint32, lambda val: int(val))
            sqlite3.register_adapter(np.uint16, lambda val: int(val))
            sqlite3.register_adapter(np.uint8, lambda val: int(val))

            # Register floating point adapters
            sqlite3.register_adapter(np.floating, lambda val: float(val))
            sqlite3.register_adapter(np.float64, lambda val: float(val))
            sqlite3.register_adapter(np.float32, lambda val: float(val))

        except ImportError:
            pass

        _ADAPTERS_REGISTERED = True


# Register adapters immediately at module import
register_sqlite_adapters()


def create_connection(
    db_path: str = "traffiq_enforcement.db",
    timeout: float = 30.0,
    check_same_thread: bool = True,
) -> sqlite3.Connection:
    """
    Create and configure a new SQLite connection with sqlite3.Row row factory
    and WAL journal mode for concurrent read-write safety.
    """
    register_sqlite_adapters()
    conn = sqlite3.connect(
        db_path,
        timeout=timeout,
        check_same_thread=check_same_thread,
    )
    conn.row_factory = sqlite3.Row

    # Performance and concurrency pragmas (ignoring errors for special in-memory URIs)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        if db_path != ":memory:" and not db_path.startswith("file:"):
            conn.execute("PRAGMA journal_mode = WAL;")
    except sqlite3.OperationalError:
        pass

    return conn


def get_connection(
    db_path: str = "traffiq_enforcement.db",
    timeout: float = 30.0,
    check_same_thread: bool = True,
) -> sqlite3.Connection:
    """
    Convenience factory to create and return a configured SQLite connection.
    """
    return create_connection(
        db_path=db_path,
        timeout=timeout,
        check_same_thread=check_same_thread,
    )


# Alias for test suite compatibility
get_db_connection = create_connection
