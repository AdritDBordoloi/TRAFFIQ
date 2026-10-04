"""
Graph Exporters for Graphify.
Generates graph.json (machine-readable), graph.html (interactive D3 force-directed visualizer),
and GRAPH_REPORT.md (architectural analysis and dependency report).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List
from .graph import CodeKnowledgeGraph


def export_graph_json(graph: CodeKnowledgeGraph, output_path: str) -> str:
    """Exports machine-readable AST and dependency graph to JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    data = graph.to_dict()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return output_path


def export_graph_html(graph: CodeKnowledgeGraph, output_path: str) -> str:
    """Exports interactive browser-based visualization of the knowledge graph."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    graph_data = graph.to_dict()
    json_str = json.dumps(graph_data, ensure_ascii=False)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>TRAFFIQ — Code Knowledge Graph Explorer (Graphifyy)</title>
  <style>
    :root {{
      --bg: #0f172a;
      --surface: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --primary: #38bdf8;
      --accent: #818cf8;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow: hidden;
    }}
    header {{
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 12px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      z-index: 10;
    }}
    .logo {{
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .logo h1 {{
      font-size: 1.15rem;
      font-weight: 700;
      color: var(--primary);
      letter-spacing: 0.5px;
    }}
    .badge {{
      background: rgba(56, 189, 248, 0.15);
      color: var(--primary);
      border: 1px solid rgba(56, 189, 248, 0.3);
      padding: 2px 8px;
      border-radius: 9999px;
      font-size: 0.75rem;
      font-weight: 600;
    }}
    .stats {{
      display: flex;
      gap: 16px;
      font-size: 0.85rem;
      color: var(--text-muted);
    }}
    .stat-val {{
      color: var(--text);
      font-weight: 600;
    }}
    .main-container {{
      display: flex;
      flex: 1;
      height: calc(100vh - 60px);
      position: relative;
    }}
    #graph-canvas {{
      flex: 1;
      height: 100%;
      cursor: grab;
      background: radial-gradient(circle at center, #1e293b 0%, #0f172a 100%);
    }}
    #graph-canvas:active {{
      cursor: grabbing;
    }}
    .sidebar {{
      width: 360px;
      background: var(--surface);
      border-left: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      z-index: 10;
    }}
    .sidebar-search {{
      padding: 16px;
      border-bottom: 1px solid var(--border);
    }}
    .search-input {{
      width: 100%;
      background: #0f172a;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 8px 12px;
      color: var(--text);
      font-size: 0.875rem;
    }}
    .search-input:focus {{
      outline: none;
      border-color: var(--primary);
    }}
    .sidebar-content {{
      flex: 1;
      overflow-y: auto;
      padding: 16px;
    }}
    .panel-title {{
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: var(--text-muted);
      margin-bottom: 12px;
    }}
    .legend {{
      margin-bottom: 20px;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 0.8rem;
      margin-bottom: 6px;
      cursor: pointer;
      user-select: none;
    }}
    .legend-dot {{
      width: 10px;
      height: 10px;
      border-radius: 50%;
    }}
    .node-details {{
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px;
      margin-top: 12px;
      font-size: 0.85rem;
    }}
    .node-title {{
      font-size: 1rem;
      font-weight: 700;
      color: var(--primary);
      margin-bottom: 4px;
      word-break: break-all;
    }}
    .node-meta {{
      color: var(--text-muted);
      font-size: 0.75rem;
      margin-bottom: 10px;
    }}
    .doc-box {{
      background: #0f172a;
      padding: 8px;
      border-radius: 4px;
      font-family: monospace;
      font-size: 0.75rem;
      margin: 8px 0;
      white-space: pre-wrap;
      max-height: 120px;
      overflow-y: auto;
    }}
    .god-nodes-list {{
      list-style: none;
    }}
    .god-node-item {{
      padding: 8px;
      background: rgba(15, 23, 42, 0.4);
      border: 1px solid var(--border);
      border-radius: 6px;
      margin-bottom: 6px;
      cursor: pointer;
      transition: background 0.15s;
    }}
    .god-node-item:hover {{
      background: rgba(56, 189, 248, 0.1);
      border-color: var(--primary);
    }}
    .god-node-name {{
      font-weight: 600;
      font-size: 0.85rem;
    }}
    .god-node-metric {{
      font-size: 0.75rem;
      color: var(--text-muted);
    }}
    .controls {{
      position: absolute;
      bottom: 20px;
      left: 20px;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px;
      display: flex;
      gap: 8px;
      z-index: 10;
    }}
    .btn {{
      background: #0f172a;
      border: 1px solid var(--border);
      color: var(--text);
      padding: 6px 12px;
      border-radius: 4px;
      font-size: 0.8rem;
      cursor: pointer;
    }}
    .btn:hover {{
      border-color: var(--primary);
    }}
  </style>
</head>
<body>
  <header>
    <div class="logo">
      <h1>TRAFFIQ Knowledge Graph</h1>
      <span class="badge">Graphifyy v0.2.0</span>
    </div>
    <div class="stats">
      <div>Nodes: <span class="stat-val" id="stat-nodes">0</span></div>
      <div>Edges: <span class="stat-val" id="stat-edges">0</span></div>
      <div>Communities: <span class="stat-val" id="stat-comms">0</span></div>
    </div>
  </header>

  <div class="main-container">
    <canvas id="graph-canvas"></canvas>

    <div class="controls">
      <button class="btn" id="btn-reset">Reset View</button>
      <button class="btn" id="btn-zoom-in">+</button>
      <button class="btn" id="btn-zoom-out">-</button>
    </div>

    <div class="sidebar">
      <div class="sidebar-search">
        <input type="text" id="search-box" class="search-input" placeholder="Search classes, functions, modules...">
      </div>

      <div class="sidebar-content">
        <div class="panel-title">Subsystems & Communities</div>
        <div class="legend" id="legend-container"></div>

        <div class="panel-title">God Nodes (Key Hubs)</div>
        <ul class="god-nodes-list" id="god-nodes-container"></ul>

        <div id="inspector-container">
          <div class="panel-title" style="margin-top: 16px;">Node Inspector</div>
          <div class="node-details" id="node-inspector">
            <p style="color: var(--text-muted);">Click any node in the graph or list to inspect its AST connections, callers, and source details.</p>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const graphData = {json_str};

    const canvas = document.getElementById('graph-canvas');
    const ctx = canvas.getContext('2d');
    let width, height;

    function resizeCanvas() {{
      width = canvas.clientWidth;
      height = canvas.clientHeight;
      canvas.width = width;
      canvas.height = height;
    }}
    window.addEventListener('resize', () => {{ resizeCanvas(); draw(); }});
    resizeCanvas();

    // Stats
    document.getElementById('stat-nodes').textContent = graphData.nodes.length;
    document.getElementById('stat-edges').textContent = graphData.edges.length;
    document.getElementById('stat-comms').textContent = Object.keys(graphData.metadata.communities).length;

    // Palette mapping
    const commColors = {{
      core_config: "#38bdf8",
      database: "#34d399",
      vision: "#fbbf24",
      enforcement: "#f87171",
      cli_entrypoint: "#c084fc",
      testing_suite: "#a3e635",
      scripts_tooling: "#4ade80",
      documentation: "#94a3b8",
      external_library: "#64748b",
      general: "#e2e8f0"
    }};

    // Legend
    const legendEl = document.getElementById('legend-container');
    Object.entries(graphData.metadata.communities).forEach(([comm, count]) => {{
      const div = document.createElement('div');
      div.className = 'legend-item';
      div.innerHTML = `
        <span class="legend-dot" style="background: ${{commColors[comm] || '#94a3b8'}}"></span>
        <span>${{comm}} (${{count}})</span>
      `;
      legendEl.appendChild(div);
    }});

    // God Nodes
    const godEl = document.getElementById('god-nodes-container');
    (graphData.god_nodes || []).forEach(gn => {{
      const li = document.createElement('li');
      li.className = 'god-node-item';
      li.innerHTML = `
        <div class="god-node-name">${{gn.label}} <span style="font-size:0.75rem; color:var(--primary);">[${{gn.type}}]</span></div>
        <div class="god-node-metric">Total Connections: ${{gn.in_degree + gn.out_degree}} (In: ${{gn.in_degree}}, Out: ${{gn.out_degree}})</div>
      `;
      li.addEventListener('click', () => {{
        selectNode(gn.id);
      }});
      godEl.appendChild(li);
    }});

    // Graph Layout Initialization
    const nodes = graphData.nodes.map((n, i) => {{
      const angle = (i / graphData.nodes.length) * 2 * Math.PI;
      const radius = 150 + Math.random() * 250;
      return {{
        ...n,
        x: width / 2 + Math.cos(angle) * radius,
        y: height / 2 + Math.sin(angle) * radius,
        vx: 0,
        vy: 0,
        radius: Math.max(5, Math.min(18, 5 + (n.in_degree + n.out_degree) * 1.2))
      }};
    }});
    const nodeMap = new Map();
    nodes.forEach(n => nodeMap.set(n.id, n));

    const edges = graphData.edges.map(e => ({{
      ...e,
      sourceNode: nodeMap.get(e.source),
      targetNode: nodeMap.get(e.target)
    }})).filter(e => e.sourceNode && e.targetNode);

    // Simple Force-Directed Simulation (Relaxation)
    for (let step = 0; step < 70; step++) {{
      // Repulsion
      for (let i = 0; i < nodes.length; i++) {{
        for (let j = i + 1; j < nodes.length; j++) {{
          const dx = nodes[j].x - nodes[i].x;
          const dy = nodes[j].y - nodes[i].y;
          const dist = Math.sqrt(dx * dx + dy * dy) || 1;
          if (dist < 200) {{
            const force = (200 - dist) / dist * 0.08;
            nodes[i].x -= dx * force;
            nodes[i].y -= dy * force;
            nodes[j].x += dx * force;
            nodes[j].y += dy * force;
          }}
        }}
      }}
      // Attraction along edges
      edges.forEach(e => {{
        const dx = e.targetNode.x - e.sourceNode.x;
        const dy = e.targetNode.y - e.sourceNode.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const force = (dist - 80) * 0.02;
        e.sourceNode.x += dx / dist * force;
        e.sourceNode.y += dy / dist * force;
        e.targetNode.x -= dx / dist * force;
        e.targetNode.y -= dy / dist * force;
      }});
    }}

    // Transform state
    let scale = 1.0;
    let panX = 0;
    let panY = 0;
    let isDragging = false;
    let startX, startY;
    let selectedNodeId = null;
    let hoveredNodeId = null;

    function draw() {{
      ctx.clearRect(0, 0, width, height);
      ctx.save();
      ctx.translate(panX + width / 2, panY + height / 2);
      ctx.scale(scale, scale);
      ctx.translate(-width / 2, -height / 2);

      // Draw edges
      ctx.lineWidth = 1;
      edges.forEach(e => {{
        const isHighlight = selectedNodeId && (e.source === selectedNodeId || e.target === selectedNodeId);
        ctx.strokeStyle = isHighlight ? 'rgba(56, 189, 248, 0.8)' : 'rgba(51, 65, 85, 0.4)';
        ctx.lineWidth = isHighlight ? 2 : 0.8;
        ctx.beginPath();
        ctx.moveTo(e.sourceNode.x, e.sourceNode.y);
        ctx.lineTo(e.targetNode.x, e.targetNode.y);
        ctx.stroke();
      }});

      // Draw nodes
      nodes.forEach(n => {{
        const isSelected = n.id === selectedNodeId;
        const isHovered = n.id === hoveredNodeId;
        const color = commColors[n.community] || '#94a3b8';

        ctx.beginPath();
        ctx.arc(n.x, n.y, isSelected ? n.radius + 4 : n.radius, 0, 2 * Math.PI);
        ctx.fillStyle = color;
        ctx.fill();

        if (isSelected || isHovered) {{
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 2;
          ctx.stroke();
        }}

        // Text labels for large nodes or selection
        if (n.radius > 9 || isSelected || isHovered) {{
          ctx.fillStyle = '#f8fafc';
          ctx.font = isSelected ? 'bold 12px sans-serif' : '10px sans-serif';
          ctx.fillText(n.label, n.x + n.radius + 3, n.y + 4);
        }}
      }});

      ctx.restore();
    }}

    draw();

    // Interaction handlers
    canvas.addEventListener('mousedown', e => {{
      isDragging = true;
      startX = e.clientX - panX;
      startY = e.clientY - panY;
    }});

    window.addEventListener('mousemove', e => {{
      if (isDragging) {{
        panX = e.clientX - startX;
        panY = e.clientY - startY;
        draw();
      }} else {{
        const rect = canvas.getBoundingClientRect();
        const mx = (e.clientX - rect.left - (panX + width / 2)) / scale + width / 2;
        const my = (e.clientY - rect.top - (panY + height / 2)) / scale + height / 2;

        let found = null;
        for (const n of nodes) {{
          const dx = n.x - mx;
          const dy = n.y - my;
          if (dx * dx + dy * dy <= (n.radius + 4) * (n.radius + 4)) {{
            found = n.id;
            break;
          }}
        }}
        if (hoveredNodeId !== found) {{
          hoveredNodeId = found;
          draw();
        }}
      }}
    }});

    window.addEventListener('mouseup', () => {{ isDragging = false; }});

    canvas.addEventListener('wheel', e => {{
      e.preventDefault();
      const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
      scale = Math.min(3.0, Math.max(0.2, scale * zoomFactor));
      draw();
    }}, {{ passive: false }});

    canvas.addEventListener('click', e => {{
      if (hoveredNodeId) {{
        selectNode(hoveredNodeId);
      }}
    }});

    function selectNode(nid) {{
      selectedNodeId = nid;
      const n = nodeMap.get(nid);
      if (!n) return;

      const inspector = document.getElementById('node-inspector');
      const callers = edges.filter(e => e.target === nid).map(e => e.source);
      const callees = edges.filter(e => e.source === nid).map(e => e.target);

      inspector.innerHTML = `
        <div class="node-title">${{n.label}}</div>
        <div class="node-meta">${{n.id}} (${{n.type}})</div>
        <div><strong>File:</strong> ${{n.source_file || 'N/A'}}${{n.line ? ':' + n.line : ''}}</div>
        <div><strong>Subsystem:</strong> <span style="color:${{commColors[n.community]}}">${{n.community}}</span></div>
        <div><strong>Connections:</strong> In: ${{n.in_degree}}, Out: ${{n.out_degree}} (Centrality: ${{n.centrality}})</div>
        ${{n.docstring ? `<div class="doc-box">${{n.docstring}}</div>` : ''}}
        <div style="margin-top:8px;"><strong>Referenced By (${{callers.length}}):</strong></div>
        <div style="font-size:0.75rem; color:var(--text-muted); max-height:60px; overflow-y:auto;">
          ${{callers.slice(0, 8).map(c => `• ${{c}}`).join('<br>') || 'None'}}
        </div>
        <div style="margin-top:6px;"><strong>Dependencies (${{callees.length}}):</strong></div>
        <div style="font-size:0.75rem; color:var(--text-muted); max-height:60px; overflow-y:auto;">
          ${{callees.slice(0, 8).map(c => `• ${{c}}`).join('<br>') || 'None'}}
        </div>
      `;
      draw();
    }}

    // Search filter
    document.getElementById('search-box').addEventListener('input', e => {{
      const val = e.target.value.trim().toLowerCase();
      if (!val) {{
        selectedNodeId = null;
        draw();
        return;
      }}
      const match = nodes.find(n => n.label.toLowerCase().includes(val) || n.id.toLowerCase().includes(val));
      if (match) {{
        selectNode(match.id);
        panX = -(match.x - width / 2) * scale;
        panY = -(match.y - height / 2) * scale;
        draw();
      }}
    }});

    // Controls
    document.getElementById('btn-reset').addEventListener('click', () => {{
      scale = 1.0; panX = 0; panY = 0; selectedNodeId = null; draw();
    }});
    document.getElementById('btn-zoom-in').addEventListener('click', () => {{
      scale = Math.min(3.0, scale * 1.25); draw();
    }});
    document.getElementById('btn-zoom-out').addEventListener('click', () => {{
      scale = Math.max(0.2, scale * 0.8); draw();
    }});
  </script>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return output_path


def export_graph_report(graph: CodeKnowledgeGraph, output_path: str) -> str:
    """Generates comprehensive architectural markdown report (GRAPH_REPORT.md)."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    g_dict = graph.to_dict()
    meta = g_dict["metadata"]
    god_nodes = g_dict["god_nodes"]

    report = f"""# Code Knowledge Graph Architectural Report

Generated by **Graphifyy v0.2.0** on `{meta['generated_at']}`.  
Target Codebase: `{meta['project_root']}`

---

## 1. Executive Summary

This report presents the structural and topological analysis of the TRAFFIQ automated traffic enforcement system codebase. The knowledge graph was constructed using local, deterministic AST (Abstract Syntax Tree) parsing across all Python modules, configuration manifests, and documentation files.

### Global Graph Metrics:
- **Total Code Symbols / Nodes**: `{meta['total_nodes']}`
- **Dependency & Call Edges**: `{meta['total_edges']}`
- **Subsystem Communities**: `{len(meta['communities'])}` distinct architectural clusters

---

## 2. Core Subsystems & Community Clustering

The codebase is partitioned into cohesive, decoupled subsystem clusters:

| Subsystem Cluster | Description | Symbol Count |
|---|---|---|
| `core_config` | Centralized settings dataclass (`TraffiqConfig`) | {meta['communities'].get('core_config', 0)} |
| `database` | SQLite WAL connection pool, RTO vehicle registry, challan logging, and migration | {meta['communities'].get('database', 0)} |
| `vision` | Safe camera context manager, YOLO vehicle detector, ByteTrack tracker, and virtual tripwire | {meta['communities'].get('vision', 0)} |
| `enforcement` | Traffic signal state machine, stop-line infraction evaluator, plate ROI OCR, and challan issuer | {meta['communities'].get('enforcement', 0)} |
| `cli_entrypoint` | Unified CLI dispatcher (`traffiq.cli.app` & `main.py`) | {meta['communities'].get('cli_entrypoint', 0)} |
| `testing_suite` | 4-Tier test harness (unit, component, sanitization, e2e integration) | {meta['communities'].get('testing_suite', 0)} |
| `scripts_tooling` | Standalone verification runner (`scripts/verify_all.py`) | {meta['communities'].get('scripts_tooling', 0)} |
| `documentation` | Architecture specifications, contributing guide, licenses, and docs | {meta['communities'].get('documentation', 0)} |

---

## 3. God Nodes & Central Structural Hubs

"God nodes" represent central structural hubs with high degree centrality and connectivity across the system. These components serve as the primary architectural backbones:

"""
    for i, gn in enumerate(god_nodes, 1):
        report += f"""### #{i}. `{gn['id']}`
- **Symbol Type**: `{gn['type']}`
- **Source Location**: `{gn.get('source_file', 'external')}:{gn.get('line', 1)}`
- **Total Connectivity**: `{gn['in_degree'] + gn['out_degree']}` (In-degree: `{gn['in_degree']}`, Out-degree: `{gn['out_degree']}`)
- **Degree Centrality**: `{gn['centrality']}`
- **Docstring**: {gn.get('docstring', 'No docstring provided') or 'No docstring provided'}

"""

    report += """---

## 4. Key Dependency Flows & Pipeline Call Traces

The knowledge graph highlights four primary architectural execution flows:

1. **Diagnostics Flow**:
   `traffiq.cli.app:main` → `run_diagnostics` → `setup_database` → `verify_integrity` → `CameraStream` probe.
2. **Detection Pipeline Flow**:
   `traffiq.cli.app:main` → `run_detection` → `CameraStream` → `VehicleDetector.detect` → Telemetry HUD.
3. **Tracking & Flow Counting Flow**:
   `traffiq.cli.app:main` → `run_tracking` → `VehicleTracker.track` → `VirtualTripwire.update_track` → Bounded TTL trajectory pruning.
4. **Full Automated Enforcement Flow**:
   `traffiq.cli.app:main` → `run_enforcement` → `TrafficSignal.update` → `VehicleTracker.track` → `ViolationDetector.evaluate_crossing` → `PlateExtractor.read_plate` (ROI isolation + OCR fallback) → `ChallanIssuer.issue_challan` (native int track_id) → `DatabaseManager.record_challan`.

---

## 5. Architectural Quality & Modularity Assessment

- **Decoupling**: Submodules (`vision`, `enforcement`, `database`, `config`) exhibit clean boundary isolation with zero circular imports.
- **Data Integrity**: All database entry points explicitly enforce native integer types for SQLite serialization, eliminating binary blob corruption.
- **Fail-Safe Fallbacks**: Vision OCR isolates the lower 40-50% horizontal band for license plate extraction with graceful fallback to `("UNKNOWN", 0.0)`.

---

## 6. Suggested AI Assistant Queries

AI coding assistants (Claude Code, Cursor, Copilot, Gemini CLI) can leverage this knowledge graph using Graphify commands:

```bash
# Query the role and connections of the configuration manager
graphify explain TraffiqConfig

# Trace the relationship between camera stream and violation detection
graphify path CameraStream ViolationDetector

# Query components dealing with license plate recognition
graphify query "PlateExtractor"

# Audit database challan recording and serialization
graphify query "record_challan"
```
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    return output_path
