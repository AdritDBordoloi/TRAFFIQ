# Contributing to TRAFFIQ

Thank you for your interest in contributing to **TRAFFIQ: AI-Powered Traffic Monitoring & Automated E-Challan System**! We welcome contributions from developers, researchers, and traffic safety enthusiasts.

This guide outlines our development workflow, architectural conventions, coding standards, and testing procedures.

---

## Table of Contents
1. [Code of Conduct](#code-of-conduct)
2. [Development Environment Setup](#development-environment-setup)
3. [Package Architecture](#package-architecture)
4. [Coding Standards](#coding-standards)
5. [Testing & Quality Assurance](#testing--quality-assurance)
6. [Pull Request Process](#pull-request-process)
7. [Licensing](#licensing)

---

## Code of Conduct

We are committed to providing a friendly, safe, and welcoming environment for all contributors regardless of background or experience level.

- **Be respectful and inclusive**: Treat fellow contributors with kindness and professional courtesy.
- **Constructive collaboration**: Focus feedback on code quality, architecture, and problem-solving.
- **Zero tolerance**: Harassment, derogatory comments, and disruptive conduct will not be tolerated.

---

## Development Environment Setup

### 1. Prerequisites
- **Python**: 3.10 or higher (tested with Python 3.10 through 3.14).
- **Operating System**: Windows 10/11 (with DirectShow camera support), Linux, or macOS.
- **Hardware (Recommended)**: NVIDIA GeForce GPU (e.g., RTX 4060 with 8GB VRAM) for accelerated real-time YOLOv8s inference. CPU fallback is fully supported for testing and development.
- **Git**: For version control.

### 2. Clone the Repository
```bash
git clone https://github.com/traffiq-org/TRAFFIQ.git
cd TRAFFIQ
```

### 3. Create & Activate Virtual Environment
Always work inside a dedicated virtual environment:

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install PyTorch with CUDA Support
If an NVIDIA GPU with CUDA 12.6 is available, install the accelerated PyTorch wheels:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

*(For CPU-only environments, standard `pip install torch torchvision` suffices).*

### 5. Install Project Dependencies
Install the package requirements and developer tooling:

```bash
pip install -r requirements.txt
pip install -e ".[dev,test]"
```

### 6. Verify Hardware Acceleration & Camera Stream
Run preflight diagnostics to ensure PyTorch and OpenCV detect your hardware:
```bash
python main.py --mode diagnostics --no-gui
```

---

## Package Architecture

TRAFFIQ is organized into a modular Python package located in the `traffiq/` directory:

```
TRAFFIQ/
├── main.py                     # Unified CLI launcher entrypoint
├── pyproject.toml              # PEP 517/621 packaging metadata
├── requirements.txt            # Project runtime & development dependencies
├── LICENSE                     # MIT License (2026)
├── CONTRIBUTING.md             # Contributor guidelines
├── README.md                   # System documentation & architectural diagrams
├── .gitignore                  # Git hygiene rules
├── traffiq/                    # Core modular library
│   ├── __init__.py             # Package metadata (__version__ = "1.0.0")
│   ├── config/                 # Centralized configuration
│   │   ├── __init__.py
│   │   └── settings.py         # TraffiqConfig dataclass & parameter validation
│   ├── database/               # Database management & RTO registry
│   │   ├── __init__.py
│   │   ├── connection.py       # SQLite connection manager & adapter registration
│   │   ├── repository.py       # Vehicle queries & challan logging (native int)
│   │   └── migration.py        # In-place blob-to-int sanitization & integrity verification
│   ├── vision/                 # Computer vision & trajectory tracking
│   │   ├── __init__.py
│   │   ├── camera.py           # Safe CameraStream context manager (try...finally)
│   │   ├── detector.py         # YOLOv8 vehicle detection (GPU/CPU fallback)
│   │   ├── tracker.py          # ByteTrack wrapper with bounded TTL trajectory cache
│   │   └── tripwire.py         # Virtual counting stop-line & directional crossing math
│   ├── enforcement/            # Signal, violation, OCR & challans
│   │   ├── __init__.py
│   │   ├── signal.py           # TrafficSignal state machine (timed cycles + manual toggle)
│   │   ├── violation.py        # Red-light stop-line crossing evaluator
│   │   ├── ocr.py              # License plate ROI isolation, CLAHE preprocessing, EasyOCR
│   │   └── challan.py          # ChallanIssuer & HUD alert rendering
│   └── cli/                    # Command-line interface dispatcher
│       ├── __init__.py
│       └── app.py              # Argparse CLI builder and pipeline runners
├── tests/                      # 4-Tier comprehensive automated test suite
│   ├── conftest.py             # Shared synthetic fixtures & test helpers
│   ├── tier1_unit/             # Fast logic unit tests
│   ├── tier2_component/        # Component tests (plate ROI, signal, OCR fallback)
│   ├── tier3_sanitization/     # Database integrity & memory bounds
│   └── tier4_e2e_integration/  # E2E CLI, repository hygiene & graphify tests
├── scripts/
│   └── verify_all.py           # Standalone zero-dependency acceptance verifier
└── graphify-out/               # Code knowledge graph outputs
```

### Module Boundaries & Responsibilities
- **`traffiq.config`**: Centralizes all runtime thresholds, hardware device selection, fine amounts, and geometry into a typed `TraffiqConfig` dataclass. Never hardcode magic numbers in vision or enforcement modules.
- **`traffiq.database`**: Interacts with SQLite (`traffiq_enforcement.db`). Strictly enforces that `track_id` values are native Python integers (`int`), never raw NumPy scalars (`np.int32` / `np.int64`).
- **`traffiq.vision`**: Manages video capture (`CameraStream`), inference (`VehicleDetector`), multi-object tracking (`VehicleTracker`), and virtual tripwires (`VirtualTripwire`). Implements TTL eviction to prevent unbounded memory growth while preserving active trajectories.
- **`traffiq.enforcement`**: Handles the traffic signal state machine, infraction detection, license plate ROI isolation (isolating the lower 40-50% horizontal band of vehicle bounding boxes), image preprocessing (grayscale, CLAHE, bilateral filtering), EasyOCR extraction with graceful fallback to `("UNKNOWN", 0.0)`, and on-screen HUD alerts.
- **`traffiq.cli`**: Parses CLI arguments and routes execution to one of the 4 operational modes: `diagnostics`, `detection`, `tracking`, or `enforcement`.

---

## Coding Standards

### 1. Code Style & Formatting
- **PEP 8**: Follow standard Python style conventions.
- **Line Length**: Keep lines under 100 characters where practical.
- **Indentation**: 4 spaces per indentation level (no tabs).

### 2. Type Annotations
- Use strict type hints for all public functions, class methods, and return values:
  ```python
  def lookup_vehicle(self, plate_number: str) -> Optional[Dict[str, str]]:
      ...
  ```
- Import standard typing types from `typing` (`Optional`, `Dict`, `List`, `Tuple`, `Any`) or use Python 3.10+ union types (`str | None`) with `from __future__ import annotations`.

### 3. Documentation & Comments
- Provide clear Google-style or Sphinx-style docstrings for all modules, classes, and public methods.
- Document inputs, outputs, exceptions raised, and algorithmic assumptions.
- Explain the *why*, not just the *what*, when implementing non-obvious logic (e.g., CLAHE tile grids, tripwire vector cross-products).

### 4. Database Serialization Rules
- **No NumPy scalar types in SQLite**: Standard `sqlite3` serializes NumPy integer scalars as raw binary `BLOB` buffers. Always cast `track_id` to native Python `int(track_id)` prior to executing `INSERT` statements.
- Any new database tables or columns must be accompanied by appropriate schema migration tests.

### 5. Resource Safety & Headless Support
- Hardware resources (OpenCV camera capture, GUI display windows) must always be wrapped in context managers (`CameraStream`) or `try...finally` blocks to guarantee resource release on termination.
- All visualization code must respect `--no-gui` (`config.no_gui == True`) so that automated CI/CD runners can execute pipelines headlessly without an X11/Windows display server.

---

## Testing & Quality Assurance

TRAFFIQ enforces a 4-tier testing strategy to maintain rock-solid reliability across pure logic, vision components, database integrity, and system integration.

### Test Tiers
1. **Tier 1 (Unit Tests)**: Fast pure-logic tests that require no GPU and perform zero disk/network I/O (`tests/tier1_unit/`).
   - Configuration defaults and validation
   - License plate text normalization regex
   - Tripwire vector crossing mathematics
   - CLI argument parsing logic
2. **Tier 2 (Component Tests)**: Component-level tests using synthetic frames and mocks (`tests/tier2_component/`).
   - Vehicle bounding box plate ROI isolation (lower 40-50% band)
   - CLAHE & bilateral preprocessing pipeline
   - Graceful fallback on unreadable/blank images returning `("UNKNOWN", 0.0)`
   - Traffic light state machine transitions and manual toggles
3. **Tier 3 (Sanitization & Integrity Tests)**: Database serialization and memory bounds (`tests/tier3_sanitization/`).
   - Strict `track_id` native `int` insertion (0 binary blobs)
   - Migration logic for existing legacy blob rows
   - Bounded trajectory cache with TTL pruning without clearing active tracks
4. **Tier 4 (E2E Integration Tests)**: Subprocess CLI execution and repository hygiene (`tests/tier4_e2e_integration/`).
   - `python main.py --help` verification across all 4 operational modes
   - Headless diagnostics execution (`--no-gui --max-frames 1`)
   - Repository documentation and Git hygiene verification
   - Graphifyy CLI integration tests

### Running the Test Suite
Run the full test suite using `pytest`:
```bash
pytest tests/ -v
```

Or using Python's standard `unittest` discovery:
```bash
python -m unittest discover tests
```

### Running Acceptance Verification
Run the zero-dependency acceptance verification runner:
```bash
python scripts/verify_all.py
```
This script validates all acceptance criteria and reports a comprehensive pass/fail summary.

---

## Pull Request Process

1. **Create a Topic Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bugfix-name
   ```
2. **Implement Changes**:
   - Write clean, well-tested code following the coding standards above.
   - Add new tests for any added functionality or fixed bugs.
3. **Verify Git Cleanliness**:
   - Ensure no binary models (`*.pt`), SQLite databases (`*.db`), virtual environments (`.venv`), or evidence snapshots (`violations/*.jpg`) are tracked in git.
   - Run `git status` to ensure `.gitignore` excludes sensitive files.
4. **Run All Tests**:
   - Confirm that all unit tests, component tests, and `scripts/verify_all.py` pass with 100% success.
5. **Submit Pull Request**:
   - Write a clear PR title and descriptive summary of changes.
   - Link any related issues.
   - Ensure CI checks pass.

---

## Licensing

By contributing to TRAFFIQ, you agree that your contributions will be licensed under the [MIT License](LICENSE).
