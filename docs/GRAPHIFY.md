# Graphifyy Knowledge Graph Integration Guide

This guide documents the installation, generation, querying, and ongoing maintenance of the codebase knowledge graph in **TRAFFIQ** using **Graphifyy** (`graphify`).

---

## 1. Overview

Graphify is an AST-driven codebase knowledge graph engine. Instead of relying on lossy vector embeddings or brute-force text search (`grep`), Graphify performs deterministic Abstract Syntax Tree (AST) parsing on Python modules, configuration manifests, and documentation files to construct a queryable property graph.

In TRAFFIQ, Graphify maps all modules (`traffiq.vision`, `traffiq.enforcement`, `traffiq.database`, `traffiq.config`, `traffiq.cli`), class hierarchies, call graphs, and dependency flows into structured knowledge representations consumed by developers and AI coding assistants.

```
Repository Source Code (traffiq/)
       │
       ▼ [Deterministic AST Parsing]
CodeKnowledgeGraph (Nodes, Edges, Communities)
       │
  ┌────┼─────────────────────────┐
  ▼    ▼                         ▼
graph.json                   graph.html                 GRAPH_REPORT.md
(Machine-Readable Graph)    (Interactive UI)          (Architectural Audit)
```

---

## 2. Installation Instructions

### Package & CLI Naming Convention
- **PyPI Distribution Package**: `graphifyy` (double 'y').
- **CLI Executable Command**: `graphify` (single 'y').

### Installing in Virtual Environment
Activate your local Python virtual environment (`.venv`) and install `graphifyy`:

```bash
# Windows PowerShell
.\.venv\Scripts\pip.exe install graphifyy

# Linux / macOS
source .venv/bin/activate
pip install graphifyy
```

Alternatively, install via project extras specified in `pyproject.toml`:

```bash
pip install -e ".[tooling]"
```

### Verifying Installation
Verify that the `graphify` CLI executable is available and outputs version information:

```bash
# Direct CLI execution
graphify --version

# Or explicit virtual environment path (Windows)
.\.venv\Scripts\graphify.exe --version

# Or via Python module execution
python -m graphifyy --version
```

Expected output:
```text
graphify 0.2.0 (graphifyy)
```

---

## 3. Knowledge Graph Generation Commands

To generate the complete code knowledge graph for the TRAFFIQ repository, run `graphify` against the repository root:

```bash
# From project root D:\Programs\TRAFFIQ
graphify .
```

Or invoke the extraction subcommand explicitly:

```bash
graphify extract . --output-dir graphify-out
```

### Execution Process
When executed, Graphify:
1. Recursively traverses all `.py` files in the repository (excluding `.venv`, `.git`, `__pycache__`, and `violations`).
2. Extracts modules, classes, methods, functions, imports, and cross-module calls into directed graph nodes.
3. Groups symbols into cohesive subsystem communities (`core_config`, `database`, `vision`, `enforcement`, `cli_entrypoint`, `testing_suite`, `documentation`).
4. Calculates network metrics (in-degree, out-degree, degree centrality) and identifies "God nodes".
5. Emits three artifacts in `graphify-out/`:
   - `graphify-out/graph.json`
   - `graphify-out/graph.html`
   - `graphify-out/GRAPH_REPORT.md`

---

## 4. Querying & Navigation Commands

Graphify provides CLI utilities to query symbols, trace dependency paths, and explain component architecture:

### 4.1 Component & Symbol Querying (`graphify query`)
Search for functions, classes, or modules matching a query string:

```bash
# Search for plate extraction components
graphify query "PlateExtractor"

# Search for database serialization routines
graphify query "record_challan"

# Search for configuration models
graphify query "TraffiqConfig"
```

Output displays the symbol type, location (`file:line`), subsystem community, centrality score, and docstring summary.

### 4.2 Relationship & Path Tracing (`graphify path`)
Find the shortest structural or call path between two components across the codebase:

```bash
# Trace connection from CameraStream to ViolationDetector
graphify path CameraStream ViolationDetector

# Trace connection from CLI main to DatabaseManager
graphify path main DatabaseManager
```

Output:
```text
[graphify] Path from 'main' to 'DatabaseManager' (3 hops):
main -> traffiq.cli.app.main -> traffiq.cli.app.run_enforcement -> traffiq.database.repository.DatabaseManager
```

### 4.3 Structural Explanation (`graphify explain`)
Synthesize a comprehensive architectural summary of any class or module, including all incoming callers and outgoing dependencies:

```bash
# Explain the central configuration dataclass
graphify explain TraffiqConfig

# Explain the ByteTrack vehicle tracker
graphify explain VehicleTracker
```

---

## 5. Description of Generated Artifacts

| Artifact | Format | Purpose & Description |
|---|---|---|
| `graphify-out/graph.json` | JSON | Machine-readable property graph containing full AST nodes, edge relations (`contains`, `imports`, `calls`, `inherits`), community IDs, centrality scores, and metadata. Used for automated queries and AI assistant GraphRAG. |
| `graphify-out/graph.html` | Interactive HTML | Standalone browser visualization. Features force-directed layout, node coloring by community, search filtering, zoom/pan controls, and an inspector sidebar showing callers/callees upon clicking any node. |
| `graphify-out/GRAPH_REPORT.md` | Markdown | Comprehensive plain-language architectural report. Details executive metrics, community breakdown, God nodes analysis, critical execution flows, and suggested assistant exploration prompts. |

---

## 6. Re-Indexing Maintenance Procedures

During ongoing software development, keep the knowledge graph synchronized with code changes:

### 6.1 Incremental Update & Re-Indexing
Whenever new modules, classes, or pipeline features are added to `traffiq/`:

```bash
# Re-index changed files and update graphify-out/
graphify update .
```

### 6.2 Pre-Commit & Verification Workflow
Verify graph availability and integrity as part of automated CI or pre-commit checks:

```bash
# Run standalone verification runner (checks Criterion 5)
python scripts/verify_all.py

# Run Tier 4 integration test suite
pytest tests/tier4_e2e_integration/test_graphify_cli.py
```

### 6.3 Clean Rebuild
To regenerate the knowledge graph from scratch:

```bash
# Remove prior artifacts and re-index
rm -r graphify-out/
graphify .
```

---

## 7. Integration with AI Coding Assistants

Graphify knowledge graphs can be integrated into AI coding tools (Claude Code, Cursor, GitHub Copilot CLI, Gemini CLI) as a structural memory layer:

1. **Context Expansion**: Assistants load `graphify-out/graph.json` to understand class hierarchies without reading all raw source files.
2. **Deterministic Navigation**: Eliminates hallucinated import paths or stale class names.
3. **Targeted Inspections**: Assistants invoke `graphify explain <symbol>` to resolve dependencies before proposing refactors.
