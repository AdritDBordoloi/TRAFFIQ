# TRAFFIQ Test Suite Readiness & Verification Manual (TEST_READY.md)

## 1. Executive Summary
The TRAFFIQ automated test suite is fully authored, structured, and ready for continuous validation across all project milestones. It implements a 4-Tier test architecture with 16 test modules and a standalone acceptance runner (`scripts/verify_all.py`) that directly validates all 6 acceptance criteria from `ORIGINAL_REQUEST.md`.

---

## 2. Test Execution Commands

### Primary Test Runner (Pytest)
Execute the complete test suite:
```bash
pytest tests/ -v
```

Execute individual tiers:
```bash
# Tier 1: Fast Unit Tests
pytest tests/tier1_unit/ -v

# Tier 2: Component & Vision Logic Tests
pytest tests/tier2_component/ -v

# Tier 3: Database Sanitization & Memory Bounds Tests
pytest tests/tier3_sanitization/ -v

# Tier 4: End-to-End System Integration Tests
pytest tests/tier4_e2e_integration/ -v
```

### Standalone Zero-Dependency Runner
Directly validates all 6 acceptance criteria without requiring `pytest`:
```bash
python scripts/verify_all.py
```
**Exit Code**: Returns `0` when all 6 criteria pass.

### Standard Library Unittest Runner
```bash
python -m unittest discover tests -v
```

---

## 3. Tier Coverage Breakdown

### Tier 1: Fast Logic Unit Tests (`tests/tier1_unit/`)
Focuses on pure mathematical functions, string parsing, dataclass defaults, and CLI argument specifications. Runs in < 1 second with zero I/O and zero hardware requirements.

| Test File | Description | Target Milestone | Test Count |
|---|---|---|---|
| `test_imports.py` | Verifies clean imports of all 18 modules under `traffiq/` without circular dependencies or syntax errors. | M1, M2, M3 | 7 tests |
| `test_config.py` | Verifies `TraffiqConfig` dataclass default values, custom overrides, type validation, and serialization. | M1 | 5 tests |
| `test_tripwire_math.py` | Verifies centroid $(c_x, c_y)$ calculations, top-to-bottom crossing math, bottom-to-top rejection, and horizontal boundary states. | M2 | 6 tests |
| `test_text_cleaning.py` | Verifies plate string normalization, uppercase folding, noise stripping, and Indian RTO regex matching (`AS01AB1234`, `DL04CA9999`, etc.). | M2 | 6 tests |
| `test_cli_parsing.py` | Verifies argparse definitions, mode choices (`diagnostics`, `detection`, `tracking`, `enforcement`), and argument flags (`--fine`, `--conf`, `--no-gui`). | M3 | 4 tests |

**Tier 1 Total**: 28 test cases.

---

### Tier 2: Component & Vision Logic Tests (`tests/tier2_component/`)
Validates individual vision processing stages, image filters, fallback behaviors, and database CRUD transactions using synthetic arrays and isolated temporary databases.

| Test File | Description | Target Milestone | Test Count |
|---|---|---|---|
| `test_plate_roi.py` | Verifies license plate ROI extraction isolates the lower 40-50% horizontal band of the vehicle, boundary clamping, and aspect ratio filtering. | M2 | 5 tests |
| `test_plate_preprocessing.py` | Verifies OpenCV image preprocessing: grayscale conversion, CLAHE contrast enhancement, and bilateral filter noise reduction. | M2 | 5 tests |
| `test_ocr_fallback.py` | Verifies graceful fallback to `("UNKNOWN", 0.0)` on blank (black/white), noisy, degraded, or non-plate crops without throwing uncaught exceptions. | M2 | 8 tests |
| `test_signal_controller.py` | Verifies traffic light state machine: initial `GREEN`, automatic timer transition between `GREEN` and `RED`, manual `'t'` toggle, and elapsed time calculation. | M2 | 5 tests |
| `test_database_crud.py` | Verifies RTO vehicle registry lookups (known plates vs. missing plates), challan record insertion, and query integrity. | M1 | 4 tests |

**Tier 2 Total**: 27 test cases.

---

### Tier 3: Data Sanitization & Memory Bounds Tests (`tests/tier3_sanitization/`)
Validates database integrity, type safety, binary blob eradication, and trajectory tracking memory bounding.

| Test File | Description | Target Milestone | Test Count |
|---|---|---|---|
| `test_db_serialization.py` | Verifies `track_id` passed as `np.int64`, `np.int32`, `np.int16`, or native `int` is strictly stored as SQLite `INTEGER`, never as `BLOB`. | M1 | 4 tests |
| `test_db_migration.py` | Verifies legacy blob unpacking utility recovers little-endian byte buffers, achieves 100% integer rows in-place, and is fully idempotent. | M1 | 4 tests |
| `test_memory_bounds.py` | Verifies vehicle tracking trajectory cache uses frame-based TTL pruning, evicts inactive tracks, and NEVER wipes active in-flight trajectories with `.clear()`. | M2 | 3 tests |

**Tier 3 Total**: 11 test cases.

---

### Tier 4: End-to-End System Integration Tests (`tests/tier4_e2e_integration/`)
Validates end-to-end command line execution, tooling integrations, and repository hygiene standards.

| Test File | Description | Target Milestone | Test Count |
|---|---|---|---|
| `test_cli_execution.py` | Verifies `python main.py --help` exits code 0, displays all 4 operational modes, and tests headless diagnostics execution. | M3 | 3 tests |
| `test_repo_hygiene.py` | Verifies presence and completeness of `LICENSE`, `CONTRIBUTING.md`, `README.md`, `requirements.txt`, `.gitignore`, and reports 0 exposed binaries in `git status`. | M5 | 4 tests |
| `test_graphify_cli.py` | Verifies graphify CLI version execution (`graphify --version`) and knowledge graph generation contract (`graphify-out/`). | M4 | 2 tests |

**Tier 4 Total**: 9 test cases.

---

## 4. Feature Checklist & Test Mapping

| Feature | Feature Description | Milestone | Primary Test Module | Status |
|---|---|---|---|---|
| **F0.1** | E2E Test Suite & Harness | M0 | `tests/**`, `scripts/verify_all.py` | **READY** |
| **F1.1** | SQLite Serialization Fix & Migration | M1 | `test_db_serialization.py`, `test_db_migration.py` | **TEST HARNESS READY** |
| **F1.2** | Centralized Config & DB Submodule | M1 | `test_config.py`, `test_database_crud.py` | **TEST HARNESS READY** |
| **F2.1** | Vision Submodule & Memory Bounds | M2 | `test_tripwire_math.py`, `test_memory_bounds.py` | **TEST HARNESS READY** |
| **F2.2** | Enforcement Submodule & OCR ROI | M2 | `test_plate_roi.py`, `test_plate_preprocessing.py`, `test_ocr_fallback.py`, `test_signal_controller.py` | **TEST HARNESS READY** |
| **F3.1** | Unified CLI Launcher | M3 | `test_cli_parsing.py`, `test_cli_execution.py` | **TEST HARNESS READY** |
| **F4.1** | Graphifyy Tooling Integration | M4 | `test_graphify_cli.py` | **TEST HARNESS READY** |
| **F5.1** | Repo Structure & Git Hygiene | M5 | `test_repo_hygiene.py` | **TEST HARNESS READY** |
| **F6.1** | Final Verification & Hardening | M6 | `scripts/verify_all.py` | **READY** |

---

## 5. Standalone Acceptance Criteria Runner (`scripts/verify_all.py`)

The runner executes all 6 acceptance criteria and displays formatted terminal output:
- **Criterion 1**: Queries `traffiq_enforcement.db` for `typeof(track_id) = 'blob'` count (must equal 0) and `typeof(track_id) = 'integer'` (must equal 100% of rows).
- **Criterion 2**: Executes plate ROI extraction (verifying lower 40-50% band) and tests OCR fallback on black blank frame (verifying `("UNKNOWN", 0.0)`).
- **Criterion 3**: Imports all 18 modules under `traffiq/` via `importlib.import_module()`.
- **Criterion 4**: Runs `python main.py --help` via `subprocess` and asserts exit code 0 and presence of `diagnostics`, `detection`, `tracking`, and `enforcement`.
- **Criterion 5**: Executes `graphify --version` and checks return code 0.
- **Criterion 6**: Checks existence of `LICENSE`, `CONTRIBUTING.md`, `README.md`, `requirements.txt`, `.gitignore` and validates `git status --porcelain` for 0 exposed `.pt`, `.db`, `.venv`, or violation images.

**Execution Command**:
```bash
python scripts/verify_all.py
```
