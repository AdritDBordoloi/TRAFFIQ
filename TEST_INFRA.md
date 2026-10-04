# TRAFFIQ Test Infrastructure & Verification Architecture

## 1. Overview & Test Philosophy
The TRAFFIQ test infrastructure provides comprehensive, opaque-box, requirement-driven verification for the TRAFFIQ automated traffic enforcement system. The test harness derives directly from specifications defined in `ORIGINAL_REQUEST.md` and `PROJECT.md`.

### Core Testing Principles
1. **Opaque-Box & Requirement-Driven**: Tests validate external behavior, public API contracts, and observable side-effects (database persistence types, return values, exit codes, file artifacts), rather than internal private variables.
2. **Progressive Testability**: Tests are stratified into distinct tiers. Each tier can be executed independently, allowing verification to progress incrementally as system components and milestones are developed.
3. **State Isolation & Clean Hermeticism**: Every test sets up its own isolated state using temporary directories (`tmp_path`) or in-memory SQLite instances (`:memory:`). Tests never mutate the operational production database (`traffiq_enforcement.db`) or leave lingering image artifacts on disk.
4. **Adversarial Edge Verification**: Beyond the happy path, tests subject the system to adversarial inputs: corrupt NumPy binary blobs in SQLite, out-of-bounds bounding boxes, pure black/white/noise image inputs to OCR, trajectory dictionary churn, and malformed CLI parameters.
5. **Dual-Mode Execution**: The test suite can be run using modern test runners (`pytest tests/`) or standard zero-dependency Python runners (`python -m unittest discover tests` and `python scripts/verify_all.py`), ensuring verification succeeds in any environment.

---

## 2. 4-Tier Test Methodology

The test suite is structured into four progressive tiers located under `tests/`:

```
tests/
├── conftest.py                     # Shared fixtures (synthetic frames, isolated databases, bounding boxes)
│
├── tier1_unit/                     # Tier 1: Pure Logic Unit Tests (Fast, zero I/O, no hardware)
│   ├── test_imports.py             # Package importability, circular dependency prevention
│   ├── test_config.py              # TraffiqConfig dataclass defaults, overrides, validation
│   ├── test_tripwire_math.py       # Directional line crossing math & centroid calculation
│   ├── test_text_cleaning.py       # Plate string normalization & Indian RTO regex matching
│   └── test_cli_parsing.py         # Argparse argument parser configuration & choices
│
├── tier2_component/                # Tier 2: Component & Vision Logic (Synthetic arrays, isolated DB)
│   ├── test_plate_roi.py           # Lower 40-50% horizontal band ROI crop extraction & clamping
│   ├── test_plate_preprocessing.py # Grayscale, CLAHE contrast enhancement, bilateral filtering
│   ├── test_ocr_fallback.py        # Robust fallback to ("UNKNOWN", 0.0) on blank/blurry/noise crops
│   ├── test_signal_controller.py   # Traffic light timer state machine & manual toggle
│   └── test_database_crud.py       # RTO registry lookup and challan record creation
│
├── tier3_sanitization/             # Tier 3: Data Sanitization, Migrations & Memory Bounds
│   ├── test_db_serialization.py    # Native Python int vs np.int64 serialization (0 binary blobs)
│   ├── test_db_migration.py        # Legacy blob unpacking and in-place database sanitization
│   └── test_memory_bounds.py       # Bounded TTL trajectory pruning without .clear() cache wiping
│
└── tier4_e2e_integration/          # Tier 4: End-to-End System Integration & Repository Hygiene
    ├── test_cli_execution.py       # python main.py --help exit code 0 and operational modes
    ├── test_repo_hygiene.py        # Open-source standard files & git status clean porcelain
    └── test_graphify_cli.py        # graphify CLI availability and code graph generation
```

### Tier Descriptions & Responsibilities

| Tier | Name | Target Scope | Execution Time | Dependencies |
|---|---|---|---|---|
| **Tier 1** | Fast Logic Unit Tests | Mathematical formulations, configuration validation, regex text parsing, CLI arguments | < 1 second | Standard library, NumPy |
| **Tier 2** | Component Tests | License plate ROI extraction, OpenCV image preprocessing, OCR fallback behavior, traffic signal transitions, SQLite queries | < 5 seconds | OpenCV, NumPy, SQLite |
| **Tier 3** | Sanitization & Memory | NumPy SQLite adapter behavior, blob migration utility, trajectory TTL cache bounds | < 3 seconds | SQLite, NumPy |
| **Tier 4** | E2E Integration | Subprocess CLI entrypoints, git status hygiene, documentation presence, graphify CLI | < 10 seconds | Subprocess, Git |

---

## 3. Feature Inventory Coverage Matrix

| Feature ID | Feature Description | Milestone | Primary Test Module | Key Assertions / Verifications |
|---|---|---|---|---|
| **F0.1** | E2E Test Suite & Harness | M0 | `tests/**`, `scripts/verify_all.py` | 100% of test suites execute cleanly; zero syntax errors. |
| **F1.1** | SQLite Serialization Fix & Migration | M1 | `test_db_serialization.py`, `test_db_migration.py` | `track_id` stored as `INTEGER`; 0 `BLOB` entries; migration recovers legacy rows. |
| **F1.2** | Centralized Config & DB Submodule | M1 | `test_config.py`, `test_database_crud.py` | `TraffiqConfig` default values; RTO lookup returns owner dict; challan record inserted. |
| **F2.1** | Vision Submodule & Memory Bounds | M2 | `test_tripwire_math.py`, `test_memory_bounds.py` | Centroid math; top-to-bottom crossing; inactive tracks evicted by TTL without calling `.clear()`. |
| **F2.2** | Enforcement Submodule & OCR ROI | M2 | `test_plate_roi.py`, `test_plate_preprocessing.py`, `test_ocr_fallback.py`, `test_signal_controller.py` | Lower 40-50% ROI extracted; CLAHE/bilateral preprocessing; fallback to `("UNKNOWN", 0.0)` on blank/noise; signal toggles. |
| **F3.1** | Unified CLI Launcher | M3 | `test_cli_parsing.py`, `test_cli_execution.py` | `main.py --help` exits code 0; modes `diagnostics`, `detection`, `tracking`, `enforcement` present. |
| **F4.1** | Graphifyy Tooling Integration | M4 | `test_graphify_cli.py` | `graphify --version` succeeds; graph knowledge extraction runnable. |
| **F5.1** | Repo Structure & Git Hygiene | M5 | `test_repo_hygiene.py` | `LICENSE` (MIT), `CONTRIBUTING.md`, `README.md`, `requirements.txt`, `.gitignore` exist; 0 `.pt`/`.db`/`.venv` files exposed. |
| **F6.1** | Final Verification & Hardening | M6 | `scripts/verify_all.py` | Standalone script verifies all 6 acceptance criteria and returns exit code 0. |

---

## 4. Acceptance Criteria Verification Matrix

| # | Acceptance Criterion | Verification Implementation | Success Condition |
|---|---|---|---|
| 1 | **Data Sanitization**: 100% of rows in `challan_records` have integer `track_id` values with zero binary blobs remaining. | `tests/tier3_sanitization/test_db_serialization.py`<br>`scripts/verify_all.py::check_database_sanitization()` | `SELECT COUNT(*) FROM challan_records WHERE typeof(track_id) = 'blob'` == 0 AND `typeof(track_id) != 'integer'` == 0. |
| 2 | **Plate Extraction & Fallback**: License plate extraction includes dedicated ROI cropping/preprocessing with graceful fallback to unknown when unreadable. | `tests/tier2_component/test_plate_roi.py`<br>`tests/tier2_component/test_ocr_fallback.py`<br>`scripts/verify_all.py::check_ocr_fallback()` | ROI crops lower 40-50% horizontal band; black/white/noise frames return `("UNKNOWN", 0.0)` without raising exceptions. |
| 3 | **Package Imports**: All modules under `traffiq/` can be imported cleanly without syntax errors, circular imports, or missing dependencies. | `tests/tier1_unit/test_imports.py`<br>`scripts/verify_all.py::check_package_imports()` | `importlib.import_module()` succeeds for all 18 package submodules without raising `ImportError` or `SyntaxError`. |
| 4 | **Unified CLI Entrypoint**: `python main.py --help` exits with code 0 and displays available operational modes. | `tests/tier4_e2e_integration/test_cli_execution.py`<br>`scripts/verify_all.py::check_cli_help()` | Subprocess exit code == 0; stdout contains `diagnostics`, `detection`, `tracking`, `enforcement`. |
| 5 | **Graphifyy Integration**: `graphify --version` executes successfully. | `tests/tier4_e2e_integration/test_graphify_cli.py`<br>`scripts/verify_all.py::check_graphify()` | Subprocess exit code == 0; version string displayed. |
| 6 | **Repository & Documentation Hygiene**: `LICENSE`, `CONTRIBUTING.md`, `README.md`, `requirements.txt`, `.gitignore` exist and git status reports 0 exposed binaries. | `tests/tier4_e2e_integration/test_repo_hygiene.py`<br>`scripts/verify_all.py::check_repo_hygiene()` | All 5 files exist and are non-empty; `git status --porcelain` contains zero matches for `*.pt`, `*.db`, `.venv`, or `violations/*.jpg`. |

---

## 5. Interface Contract Verification Specifications

### 5.1 `traffiq.config.settings.TraffiqConfig`
- Must be a dataclass with default fields:
  - `camera_index`: `0` (or str for video path)
  - `frame_width`: `1280`, `frame_height`: `720`
  - `device`: `'0'` (CUDA GPU) or `'cpu'`
  - `model_path`: `'yolov8s.pt'`
  - `confidence_threshold`: `0.25`
  - `tripwire_ratio`: `0.55`
  - `fine_amount`: `1000`
  - `signal_duration`: `8.0`
  - `output_dir`: `'violations'`
  - `db_path`: `'traffiq_enforcement.db'`
  - `no_gui`: `False`
  - `max_frames`: `None`
- Must support custom overriding during instantiation and serialization to dictionary.

### 5.2 `traffiq.database` Operations
- Connection factory: registers adapter converting `numpy.integer` (`np.int64`, `np.int32`, `np.int16`) to standard Python `int`.
- `lookup_vehicle(plate_number: str) -> Optional[Dict[str, Any]]`:
  - Returns `{"owner": str, "phone": str, "model": str, "city": str, "registered": True}` for known plates.
  - Returns `None` (or `{"registered": False, ...}`) for unknown plates.
- `record_challan(track_id: int, plate_number: str, owner_name: str, violation_type: str, fine_amount: int, evidence_path: str) -> int`:
  - Enforces `isinstance(track_id, (int, np.integer))` and casts to native `int`.
  - Persists record into `challan_records`.
  - Returns generated `challan_id` primary key.
- `migrate_legacy_blobs(db_path: str) -> int`:
  - Queries rows where `typeof(track_id) = 'blob'`.
  - Decodes little-endian byte buffers to signed integers.
  - Updates rows in-place; returns count of migrated records.
- `verify_integrity(db_path: str) -> Tuple[int, int, int]`:
  - Returns `(total_rows, integer_rows, blob_rows)`.

### 5.3 `traffiq.vision` & `traffiq.enforcement`
- `Tripwire`:
  - Evaluates crossing using centroid coordinates $(c_x, c_y)$.
  - Top-to-bottom crossing: `prev_y < tripwire_y <= curr_y` -> `True`.
  - Bottom-to-top crossing: `prev_y > tripwire_y >= curr_y` -> `False`.
- `VehicleTracker`:
  - Tracks vehicle centroids across frames.
  - `prune_inactive(current_frame: int, ttl: int = 30)`:
    - Evicts entries unseen for $> \text{ttl}$ frames.
    - NEVER invokes `.clear()` on the active trajectory cache.
- `PlateExtractor`:
  - `extract_roi(vehicle_crop: np.ndarray) -> np.ndarray`:
    - Isolates lower 40-50% horizontal region of the vehicle bounding box.
    - Clamps coordinates to valid crop boundaries `[0, H]` and `[0, W]`.
  - `preprocess(roi: np.ndarray) -> np.ndarray`:
    - Grayscale conversion -> CLAHE contrast enhancement -> Bilateral noise filtering.
  - `read_plate(vehicle_crop: np.ndarray) -> Tuple[str, float]`:
    - Returns `(plate_text, confidence)`.
    - Returns `("UNKNOWN", 0.0)` on blank, noisy, or unreadable inputs without raising exceptions.
- `TrafficSignalController`:
  - Automatic timer cycling between `"GREEN"` and `"RED"`.
  - Manual toggle `'t'` immediately switches state and resets timer.

---

## 6. How to Run the Tests

### Option A: Standalone Zero-Dependency Runner (Recommended for immediate verification)
Runs all 6 acceptance criteria checks using standard library and installed dependencies:
```bash
python scripts/verify_all.py
```
Exit code `0` indicates 100% criteria passed.

### Option B: Pytest Test Runner
Runs the full 4-tier suite:
```bash
pytest tests/ -v
```

Run specific tiers:
```bash
pytest tests/tier1_unit/ -v
pytest tests/tier2_component/ -v
pytest tests/tier3_sanitization/ -v
pytest tests/tier4_e2e_integration/ -v
```

### Option C: Python Standard Library Unittest
Runs tests without requiring pytest:
```bash
python -m unittest discover tests -v
```
