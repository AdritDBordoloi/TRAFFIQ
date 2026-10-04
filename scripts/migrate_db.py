"""
Standalone database migration and verification script for TRAFFIQ.
Migrates legacy BLOB track_id entries in challan_records to native INTEGERs.
"""

from __future__ import annotations

import os
import sys

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from traffiq.database.migration import migrate_legacy_blobs, verify_integrity


def main() -> int:
    db_path = os.path.join(REPO_ROOT, "traffiq_enforcement.db")
    if not os.path.exists(db_path):
        print(f"[ERROR] Database file not found at: {db_path}", file=sys.stderr)
        return 1

    print(f"[*] Analyzing database integrity for: {db_path}")
    total_before, int_before, blob_before = verify_integrity(db_path)
    print(f"[*] Pre-migration state: {total_before} total rows, {int_before} integer rows, {blob_before} blob rows.")

    if blob_before > 0:
        print(f"[*] Migrating {blob_before} binary blob rows...")
        migrated = migrate_legacy_blobs(db_path)
        print(f"[+] Successfully migrated {migrated} rows in-place.")
    else:
        print("[+] No binary blob rows found. Database is already clean.")

    total_after, int_after, blob_after = verify_integrity(db_path)
    print(f"[*] Post-migration state: {total_after} total rows, {int_after} integer rows, {blob_after} blob rows.")

    if blob_after == 0 and (total_after == 0 or int_after == total_after):
        print("[SUCCESS] 100% of rows have integer track_id values with 0 binary blobs remaining.")
        return 0
    else:
        print(f"[FAILURE] Sanitization incomplete: {blob_after} blob rows remaining!", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
