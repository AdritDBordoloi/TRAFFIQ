# Project: TRAFFIQ — Automated Traffic Enforcement System Hardening

## Architecture
The TRAFFIQ system is decomposed into a modular Python package (`traffiq/`) with decoupled submodules, a unified CLI launcher (`main.py`), tooling integrations, and a 4-tier E2E testing harness:

```
TRAFFIQ/
├── main.py                     # Unified CLI entrypoint (traffiq.cli.app:main)
├── pyproject.toml              # Modern package configuration
├── requirements.txt            # Python dependencies (runtime + tooling + testing)
├── LICENSE                     # MIT License (2026)
├── CONTRIBUTING.md             # Contributor guidelines and architecture standards
├── README.md                   # System documentation, diagrams, usage, and hardware spec
├── .gitignore                  # Git hygiene rules
├── traffiq/
│   ├── __init__.py             # Package root metadata (__version__ = "1.0.0")
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py         # TraffiqConfig centralized configuration dataclass
│   ├── database/
│   │   ├── __init__.py
│   │   ├── connection.py       # SQLite connection manager, adapter registration
│   │   ├── repository.py       # RTO registry queries, challan recording with native int
│   │   └── migration.py        # SQLite blob-to-int sanitization & verification
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── camera.py           # Safe CameraStream context manager (try...finally)
│   │   ├── detector.py         # YOLO vehicle detector with CPU/CUDA device handling
│   │   ├── tracker.py          # ByteTrack wrapper with bounded TTL trajectory cache
│   │   └── tripwire.py         # Virtual counting tripwire & directional crossing
│   ├── enforcement/
│   │   ├── __init__.py
│   │   ├── signal.py           # Traffic signal state machine (Auto + manual toggle)
│   │   ├── violation.py        # Stop-line red light infraction evaluator
│   │   ├── ocr.py              # License plate ROI isolation, preprocessing & EasyOCR
│   │   └── challan.py          # Challan generator, fine rules & HUD banner alerts
│   └── cli/
│       ├── __init__.py
│       └── app.py              # Argparse CLI dispatcher (diagnostics, detection, tracking, enforcement)
├── tests/                      # Dual-mode automated test suite (pytest + unittest)
│   ├── conftest.py             # Shared fixtures and synthetic test frames
│   ├── tier1_unit/             # Fast logic unit tests (imports, config, math, parsing)
│   ├── tier2_component/        # Component tests (plate ROI, preprocessing, OCR fallback, signal)
│   ├── tier3_sanitization/     # Database integrity & memory bounds (blob sanitization, TTL cache)
│   └── tier4_e2e_integration/  # End-to-end integration (CLI execution, repo hygiene, graphify)
├── scripts/
│   └── verify_all.py           # Standalone zero-dependency verification runner
└── graphify-out/               # Generated code knowledge graph (graph.json, graph.html, report)
```

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | F0.1: E2E Test Suite | 4-Tier test suite (Tiers 1-4) & standalone verification script | M0 | ORIGINAL_REQUEST §Acceptance Criteria |
| 2 | F1.1: SQLite Serialization Fix & Migration | Cast track_id to native int, register adapters, migrate legacy blobs in `challan_records` | M1 | ORIGINAL_REQUEST §R1 |
| 3 | F1.2: Database Submodule & Config | `traffiq.database` and `traffiq.config` centralized settings dataclass | M1 | ORIGINAL_REQUEST §R1, §R2 |
| 4 | F2.1: Vision Submodule | `traffiq.vision` (safe CameraStream, YOLO detector, TTL trajectory tracker, tripwire) | M2 | ORIGINAL_REQUEST §R1, §R2 |
| 5 | F2.2: Enforcement Submodule & OCR ROI | `traffiq.enforcement` (plate ROI isolation, CLAHE/bilateral preproc, fallback, signal, challan) | M2 | ORIGINAL_REQUEST §R1, §R2 |
| 6 | F3.1: Unified CLI Entrypoint | `main.py` CLI launcher supporting diagnostics, detection, tracking, enforcement modes | M3 | ORIGINAL_REQUEST §R2 |
| 7 | F4.1: Graphifyy Integration | Install `graphifyy` into `.venv`, generate knowledge graph `graphify-out/`, document commands | M4 | ORIGINAL_REQUEST §R3 |
| 8 | F5.1: Repo Structure & Hygiene | `LICENSE` (MIT), `CONTRIBUTING.md`, `README.md`, `requirements.txt`, `.gitignore` hardening | M5 | ORIGINAL_REQUEST §R4 |
| 9 | F6.1: Final Verification & Hardening | Pass 100% E2E tests across Tiers 1-4 and execute Tier 5 adversarial hardening | M6 | ORIGINAL_REQUEST §Acceptance Criteria |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M0 | E2E Testing Track | Test harness infra, Tiers 1-4 test suites, `scripts/verify_all.py`, `TEST_READY.md` | none | DONE |
| M1 | Database & Config Package | `traffiq.config`, `traffiq.database`, blob-to-int migration, serialization fix | none | DONE |
| M2 | Vision & Enforcement Package | `traffiq.vision`, `traffiq.enforcement`, plate ROI extraction, memory TTL tracker | M1 | DONE |
| M3 | Unified CLI Launcher | `main.py` & `traffiq.cli.app` supporting 4 operational modes with flags | M1, M2 | DONE |
| M4 | Graphifyy Tooling Integration | Install `graphifyy`, execute `graphify .`, verify `graphify-out/` outputs | M2, M3 | DONE |
| M5 | Documentation & Git Hygiene | MIT `LICENSE`, `CONTRIBUTING.md`, updated `README.md`, `.gitignore`, `requirements.txt` | M3, M4 | DONE |
| M6 | Final Milestone & Hardening | Pass 100% of E2E test suite (Tiers 1-4), Tier 5 adversarial hardening | M0, M1-M5 | DONE |

## Interface Contracts
### `traffiq.config` ↔ All Modules
- `TraffiqConfig` dataclass:
  - `camera_index`: int or str (default 0)
  - `frame_width`: int (default 1280), `frame_height`: int (default 720)
  - `device`: str ('0' for CUDA GPU, 'cpu' for CPU)
  - `model_path`: str (default "yolov8s.pt"), `confidence_threshold`: float (default 0.25)
  - `tripwire_ratio`: float (default 0.55), `fine_amount`: int (default 1000)
  - `signal_duration`: float (default 8.0)
  - `output_dir`: str (default "violations"), `db_path`: str (default "traffiq_enforcement.db")
  - `no_gui`: bool (default False), `max_frames`: Optional[int] (default None)

### `traffiq.database` ↔ `traffiq.enforcement`
- `DatabaseManager`:
  - `get_connection()`: sqlite3.Connection with row_factory and registered `numpy.integer -> int` adapter.
  - `lookup_vehicle(plate_number: str) -> Optional[Dict[str, str]]`: Returns owner dict or None.
  - `record_challan(track_id: int, plate_number: str, owner_name: str, violation_type: str, fine_amount: int, evidence_path: str) -> int`: Strictly validates `isinstance(track_id, int)` and inserts row.
  - `migrate_legacy_blobs() -> int`: Converts existing binary blobs to native integers.
  - `verify_integrity() -> Tuple[int, int, int]`: Returns `(total_rows, integer_rows, blob_rows)`.

### `traffiq.vision` ↔ `traffiq.enforcement`
- `VehicleDetector`:
  - `detect(frame: np.ndarray) -> List[Detection]`: returns bounding boxes, confidences, class IDs.
- `VehicleTracker`:
  - `update(detections, frame) -> List[TrackedVehicle]`: returns track_id (int), bbox, centroid.
  - `prune_inactive(current_frame: int, ttl: int = 30)`: evicts tracks unseen for > ttl frames, never wiping active trajectories.
- `PlateExtractor`:
  - `extract_roi(vehicle_crop: np.ndarray) -> np.ndarray`: isolates lower 40-50% horizontal band.
  - `preprocess(roi: np.ndarray) -> np.ndarray`: grayscale + CLAHE + bilateral filter.
  - `read_plate(vehicle_crop: np.ndarray) -> Tuple[str, float]`: returns `(plate_text, confidence)` with fallback to `("UNKNOWN", 0.0)`.

## Code Layout
- Package source: `traffiq/`
- CLI Entrypoint: `main.py`
- Test suite: `tests/`
- Standalone verification script: `scripts/verify_all.py`
- Documentation: `README.md`, `CONTRIBUTING.md`, `LICENSE`, `TEST_INFRA.md`, `TEST_READY.md`
- Tooling: `graphify-out/`
