# TRAFFIQ Cleanup, Bloat-Reduction & Debug Report

- **Date**: 2026-11-02 · **Branch**: `cleanup/agents-20261006` · **Commits made**: none (everything is in the working tree)
- **Scope**: audit → remove dead code/duplication/unused deps → debug → regression-test → document
- **Tools used**: `ruff` 0.17 and `vulture` 2.16 (installed to a scratch folder, **not** into `.venv`), pytest, a mutation check, and a live run of the dashboard engine on `uploads/Ambulance_Running_on_Road…webm` with the real YOLO models.

---

## 1. Result at a glance

| Metric | Before | After |
|---|---|---|
| Test suite | 133 passed | **155 passed** (+22 regression tests), identical under `pytest` and `python -m unittest discover` |
| Lint (`ruff` F, E9, B-subset) | 66 findings | **0** |
| `scripts/verify_all.py` | 6/6, import check hard-coded to 19 modules | 6/6, **26 modules auto-discovered** (7 were never checked) |
| Genuine bugs fixed | – | **20** (1 critical startup crash, 3 high, 5 medium, 11 low — see §3) |
| Duplicate logic removed | ambulance helpers in 2 files, 3 directory trees in 3 docs, 2 module lists, 1 config whitelist | one shared module / single source of truth each |
| Declared-but-unused dependencies | Pillow, PyYAML, pytest-cov | removed (Pillow/PyYAML still arrive transitively via `ultralytics`) |
| Undeclared dependency | Flask (imported by `traffiq.web`) | declared |
| Doc size | README 21.7 KB, CONTRIBUTING 12.1 KB, PROJECT 8.3 KB | README 22.4 KB (gained the missing dashboard section + accurate tree), CONTRIBUTING 9.4 KB, PROJECT 5.8 KB → **net −4.6 KB** |

**Honest note on size.** This was mostly a *correctness and de-duplication* win, not a big line-count win. The large root-script deletions (`live_*.py`, `benchmark.py`, …, ~1,400 lines) were already in the working tree from the earlier cleanup and are not counted here. In files I touched: `live_unified.py` 1,113 → 1,027, `connection.py` 153 → 97, `settings.py` 197 → 172, `verify_all.py` 359 → 318. `engine.py` *grew* 619 → 661 because it gained real behaviour (a proper violation path, evidence-URL helper, constants) while losing a 37-line duplicate helper. `traffiq/` overall is +78 lines (5,165 → 5,243), including the new 115-line shared `ambulance.py` that replaced two ~45-line copies and ~120 lines of new behaviour/fixes.

---

## 2. Context grounding — what was read and which constraints were honoured

Read in full: `README.md`, `PROJECT.md`, `AUDIT_REPORT.md`, `pyproject.toml`, `requirements.txt`, CI workflow, every module I changed. `CONTRIBUTING.md`, `TEST_INFRA.md`, `TEST_READY.md`, `docs/GRAPHIFY.md` were read via targeted search for constraints and references (not line by line).

| Documented constraint | How it was respected |
|---|---|
| `PROJECT.md` is the **interface contract** cited by tests (`test_config`, `test_cli_parsing`, `test_plate_preprocessing`) | Not deleted. Contract sections untouched; only the duplicate file tree was replaced by a pointer and a post-M6 module note was added. |
| `track_id` must be a native `int` (CONTRIBUTING §4) | New live path goes through `ChallanIssuer`, which casts; regression test asserts SQLite `typeof(track_id)='integer'`. |
| No magic numbers in vision/enforcement (CONTRIBUTING) | New thresholds are named constants (`AMBULANCE_IOU_THRESHOLD`, `EMERGENCY_HOLD_SECONDS`, `MAX_CHALLAN_LIMIT`, …). |
| Resources in context managers / `try…finally`, honour `--no-gui` | Emergency-log DB connection now closed in `finally`; `--no-gui` paths untouched. |
| AUDIT MASTER-22: `graphifyy/` shim retained for `test_graphify_cli.py` | Kept. |
| New tables/columns need migration tests | No schema changes were made. |

---

## 3. Bugs found and fixed

Severity is my judgement. "Verified" says how I know the fix works.

| # | Sev | Where | Problem | Fix | Verified |
|---|---|---|---|---|---|
| 1 | **Critical** | `web/engine.py` ctor | `torch.device("0")` raises `Invalid device string`. `run_dashboard.py` (default `--device 0`) **crashed at startup on every CUDA machine**; CPU-only CI never hit it. Found only by the live run. | Map `"0"` → `torch.device("cuda:0")`, CPU fallback | 2 tests (mock CUDA on/off) + live run |
| 2 | High | `web/engine.py` | **Real violations were recorded through the *demo* path**: fake `track_id` (`800+n`), a *drawn* placeholder image instead of the real crop, and when the plate was unreadable it picked a **random registered owner from a hardcoded sample list** (a real violation could be charged to an unrelated person; unknown owners were invented as "Rajesh Sharma"). `ChallanIssuer` was imported but never used. | New `record_violation()` → `ChallanIssuer` (real track id, real evidence crop, `UNKNOWN` plate stays `UNKNOWN`, owner `UNKNOWN / OUT-OF-STATE`) | 4 tests + live run (`track 31`, real JPEG served) |
| 3 | High | `web/engine.py` | Dashboard **"Vehicles" KPI was always 0**: it read `VehicleTracker.counted_ids`, a set nothing ever populates. | Engine counts unique tracks it sees; removed the dead tracker sets | Live run: 8 → 91 over the clip; reset test |
| 4 | High | `web/engine.py` | **Emergency corridor** was started with a fake id (`990+n`), so the "ambulance cleared exit line" check could never match the real vehicle (only the 20 s timeout ended it). Overlap test was `inter > 0` (**1 pixel** promoted any car to "ambulance"). After clearing, the next detection **re-triggered** the corridor (flapping). | Real track id; shared match rule (IoU>0.20 ∨ IoMin>0.40 ∨ centroid-in-box, restricted to Car/Bus/Truck, same as `live_unified.py`); promote once per track | 6 unit tests + live run |
| 5 | Medium | `web/engine.py` | Identified ambulances were **ticketed** for crossing on red; `live_unified.py` exempts them. | `is_enforceable_crossing()` exemption | unit test (see limitation in §8) |
| 6 | Medium | `adaptive_signal.py` | Queue discharge ran **once per `update()` call**, ignoring `vehicle_clearance_rate` (a computed `cleared` value was thrown away). At 30 FPS it drained ~50× too fast, inflating `served_count` and the efficiency KPI. | Time-based discharge with a fractional accumulator | **Mutation check**: with the old logic the test fails (`served=[50, 37]` vs `[11, 11]` fixed) |
| 7 | Medium | `web/engine.py`, `web/server.py` | Evidence URLs came out as `/evidence/../../../../Adrit Uddipon Bordol/…` (404) because `ChallanIssuer` stores `realpath` paths while the evidence dir was unresolved (symlinks / Windows 8.3 short names). My first engine test passed *by coincidence*; the E2E smoke test exposed it. | One `evidence_url()` helper resolving both sides, refusing paths outside the root | unit test + E2E (HTTP 200 on the image) |
| 8 | Medium | `.github/workflows/ci.yml` | Step ran `tests/test_adversarial_challenger.py` (moved to tier 4 in commit `9599c7d`) → **path no longer exists**; it was also redundant with the Tier 4 step. `libgl1-mesa-glx` does not exist on Ubuntu 24.04 (`ubuntu-latest`). | Removed the step; `libgl1` | Reasoned only — I cannot run GitHub Actions here |
| 9 | Medium | `requirements.txt`, `pyproject.toml` | **Flask undeclared** though `traffiq.web` imports it → clean installs break the dashboard; web tests silently *skip* on `ImportError`. | Added `Flask>=2.2.0` | toml parses; tests import it |
| 10 | Low-Med | `web/server.py` `/api/challans` | `?limit=-1` is "unlimited" in SQLite (dumps the table); every DB error silently swallowed → `[]`; connection leaked on error. | Clamp 1..500, `finally: close()`, warning logged | test |
| 11 | Low | `web/server.py` | `POST /api/demo/ambulance {"duration":"x"}` or `NaN` → HTTP 500. | Validate, clamp to 1–120 s | test |
| 12 | Low | `web/server.py` | `app.config["JSON_SORT_KEYS"]` is **ignored by Flask ≥ 2.3**; keys were being sorted. | `app.json.sort_keys = False` | test + E2E key order |
| 13 | Low | `web/engine.py` | `reset_stats()` zeroed counters but not the id memory, so counters jumped straight back. | Clears the seen-id set | test |
| 14 | Low | `web/engine.py` | Emergency DB write held the global lock during I/O, used bare `except: pass`, leaked the connection on error. | Write outside lock, `finally: close()`, warning logged | tests |
| 15 | Low | `web/engine.py` | An OCR/DB exception while recording a violation would **kill the capture thread** silently (frozen stream). | Per-violation `try/except` with a warning | live run (thread alive 60 s) |
| 16 | Low | `vision/four_way.py` | `emergencies` dict was `int` zeros annotated `bool`; detector failures swallowed with no trace at 30 FPS. | `False` values; first failure reported once | existing tests |
| 17 | Low | `web/engine.py` | Demo challans were indistinguishable from real ones in a table that issues fines. | `violation_type` / `vehicle_type` suffixed `(Demo)`, demo receipt marked | test |
| 18 | Low | `cli/app.py` | `--priority` / `--topology` help promised features that are **not wired in**; diagnostics always printed "ALL CRITICAL SUBSYSTEMS OPERATIONAL (0 ERRORS)" even after `[WARN]` lines. | Honest help text; banner reports warning count | smoke run |
| 19 | Low | `verify_all.py`, `tier1/test_imports.py` | Hand-maintained module lists went stale (7 new modules unchecked); unused `PACKAGE_MODULES` constant; dead `ImportError` fallback branch. | `pkgutil.walk_packages` discovery; fallback removed | 26 modules checked |
| 20 | Low | `.gitignore` | WAL mode creates `*.db-shm`, which was not ignored. | Added | – |

---

## 4. Removed / simplified

| What | Where | Why safe |
|---|---|---|
| Duplicated ambulance helpers (`get_letterbox_transformer`, `detect_ambulances_oiv7`, `compute_box_iou`, `compute_box_iomin`, `point_in_box`) | `live_unified.py` (−86 lines) and `web/engine.py` (−37) → new `traffiq/vision/ambulance.py` | Bodies were functionally identical (diffed); `live_unified.py` import-smoke-tested |
| `ConnectionPool`, `db_transaction` | `database/connection.py` | Zero references in code, tests, docs |
| `TraffiqConfig.copy_with` | `config/settings.py` | Zero references |
| Hard-coded `valid_keys` whitelist | `config/settings.py` | Now `fields(cls)`; a new field can no longer be silently dropped |
| `VehicleTracker.counted_ids`, `.violated_ids`, `max_history` | `vision/tracker.py` | Never populated; all 3 call sites use keyword args |
| Dead controller state (`last_green_end`, `total_wait_time`, `cleared`, `fixed_avg_cycle`) | `adaptive_signal.py` | Written, never read |
| Dead private attrs `_single_path`, `_sources_dict` | `four_way.py` | Never read |
| Dead `ImportError` fallback + 3 unused colour constants | `scripts/verify_all.py` | Package always exists |
| 47 unused imports / placeholder-less f-strings (ruff autofix) plus the unused locals | package, tests, scripts, graphify | Each autofix-only diff reviewed: pure import removals |
| Stale `ORIGINAL_REQUEST.md` / "18 modules" wording, stale `live_violations.py` docstring reference | `verify_all.py`, `README.md`, `live_unified.py` | Referenced files don't exist |
| Duplicate directory trees | `CONTRIBUTING.md`, `PROJECT.md` | Now point to README's single, corrected tree |
| Redundant/broken CI step | `ci.yml` | See bug 8 |
| Unused deps | `Pillow`, `PyYAML`, `pytest-cov` | No direct import / `--cov` use anywhere |
| `test_ambulance.py` | repo root (tracked) | One-off model probe. **Note:** it was already gone from disk when my `rm` ran (see §9) |

## 5. Added

- `traffiq/vision/ambulance.py` – single source of truth for the OIV7 detector and match rule.
- 22 tests: `test_dashboard_engine.py` (11), `test_ambulance_matching.py` (6), controller (1), web API (3), import discovery (1).
- `.cursorignore`, `.aidigestignore` (see §7). README sections: *Web Dashboard & Standalone Pipeline* and an accurate *Repository Structure* (the dashboard and `live_unified.py` were previously undocumented).

## 6. Dependency changes

`requirements.txt` / `pyproject.toml`: **+** `Flask>=2.2.0`; **−** `Pillow`, `PyYAML`, `pytest-cov`. `graphifyy` and `pytest` stay in `requirements.txt` because CI installs only that file.

## 7. AI-context exclusions (instead of deleting)

`.cursorignore` and `.aidigestignore` exclude: `.venv/`, caches, `graphify-out/` (2.5 MB generated JSON/HTML), `uploads/` (18 MB), `violations/`, `*.pt` (≈52 MB), `*.db*`, images/video, and the historical reports `AUDIT_REPORT.md`, `CLEANUP_REPORT.md`, `TEST_INFRA.md`, `TEST_READY.md` (≈54 KB that duplicate README/CONTRIBUTING). Claude Code itself already honours `.gitignore` for search, which covers the same data directories. `graphify-out/` was regenerated (1,395 nodes) so it no longer describes deleted code.

## 8. Deliberately NOT changed

- **`live_unified.py` is not dead** — it is the actively developed flagship (the last two commits fix it) with features the package lacks (audio alert, emergency-text OCR, file picker). I only deduplicated pure helpers. It still re-implements DB init / challan / owner lookup / plate cleaning and has **no tests**; refactoring it blind (needs camera, GPU, Windows dialogs) was too risky. Recommended follow-up with tests first.
- **Unreferenced but plausible public API kept**: `DatabaseManager.add_vehicle/get_all_challans/get_challan_count/get_challan_by_id/get_challans_by_plate`, `TraffiqConfig.from_env`, `AdaptiveIntersectionController.add_vehicles`, `TrajectoryCache.get_last_position`, `TrafficSignal.bgr_color`, `CameraStream.actual_size`, `Detection.area`. Zero callers; I treated them as API surface. Say the word and I'll delete them.
- **`graphify/` + `graphifyy/`**: retained per AUDIT MASTER-22.
- **`TEST_INFRA.md`, `TEST_READY.md`, `AUDIT_REPORT.md`**: ignored for AI, not deleted. `AUDIT_REPORT.md` is untracked (deletion would be irreversible) and is partly inaccurate (describes the root scripts as retained "delegators"; they are deleted).
- **`yolov8n.pt` (6.5 MB)** is referenced nowhere; git-ignored, so I left it for you to decide.
- **Known limitation of the ambulance exemption**: in the test clip the OIV7 model identified the ambulance ~3 s *after* it crossed the line, so it was still fined once (live run). The exemption only helps when identification precedes crossing. Mitigations (delay issuing, void on late identification) are a product decision.

## 9. Open issues needing your decision

1. **`AdaptiveIntersectionController`, `FourWayVideoProcessor`, `--priority`, `--topology` are implemented and unit-tested but not wired into the CLI or dashboard.** README now says so. Wire or remove?
2. **`efficiency_gain_pct`** (`15 + 1.2 × vehicles_served`, capped at 65) in the adaptive controller is a made-up formula presented as a measured KPI.
3. **Dashboard has no auth**: `/api/demo/*`, `/api/reset`, `/api/signal/toggle` are open, and `run_dashboard.py --host 0.0.0.0` exposes them to the LAN. Also existing demo rows in your DB predate the `(Demo)` label.
4. **Working tree was modified by another process during this session.** `test_ambulance.py` and `test_camera.py` disappeared from disk without my action (several `claude.exe` processes were running, so possibly another session). I did not touch `test_camera.py`. Worth checking you don't have two sessions editing this branch.
5. No `.gitattributes`: git warns LF→CRLF on several files. I preserved each file's existing line endings.
6. Env note: `pip check` reports `easyocr requires opencv-python-headless` (venv has `opencv-python`). Pre-existing, unrelated to this work.

## 10. Verification log

- `ruff check --select F,E9,B006,B008,B017` → clean on `traffiq scripts tests graphify` and all root scripts.
- `pytest tests` → **155 passed**; `python -m unittest discover tests` → **155 OK**.
- `python scripts/verify_all.py` → 6/6 PASS (26 modules).
- Mutation check: re-injected the original per-call discharge → regression test fails (`served=[50, 37]`).
- End-to-end smoke: real `DashboardEngine` behind the real Flask app: live violation → `/api/challans` → `/evidence/…jpg` returns 200; demo endpoints; key order.
- **Live run** (real YOLO + OIV7 on GPU, ambulance clip, temp DB): thread alive 60 s, 4–22 FPS, Vehicles 0 → 91, corridors started from real track ids 31/318, real violation (track 31, `UNKNOWN` plate) with servable evidence.
- `live_unified.py` imports cleanly (`__main__` guarded) — **its runtime behaviour was not exercised** (no interactive window/file dialog here).
- **Not verified**: the GitHub Actions run itself.
