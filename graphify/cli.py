"""
Command Line Interface (CLI) for Graphify.
Provides commands:
  - graphify .               : Generates graphify-out/{graph.json, graph.html, GRAPH_REPORT.md}
  - graphify update .        : Re-indexes and updates the knowledge graph
  - graphify query "<term>"  : Searches code symbols and relationships
  - graphify path "A" "B"    : Traces path between two components
  - graphify explain "node"  : Synthesizes structural context for a component
  - graphify --version       : Outputs version information
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional, Sequence

from . import __version__
from .ast_parser import ASTCodeParser
from .graph import CodeKnowledgeGraph
from .exporters import export_graph_json, export_graph_html, export_graph_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graphify",
        description="Graphify: AST-Driven Codebase Knowledge Graph Generator & AI Assistant Tooling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"graphify {__version__} (graphifyy)",
        help="Show program's version number and exit",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: build / extract / '.'
    extract_p = subparsers.add_parser("extract", help="Extract knowledge graph from repository")
    extract_p.add_argument("directory", nargs="?", default=".", help="Target repository directory (default: current directory)")
    extract_p.add_argument("-o", "--output-dir", default="graphify-out", help="Output directory for generated graph artifacts")

    # Command: update
    update_p = subparsers.add_parser("update", help="Update knowledge graph for changed files")
    update_p.add_argument("directory", nargs="?", default=".", help="Target repository directory")
    update_p.add_argument("-o", "--output-dir", default="graphify-out", help="Output directory")

    # Command: query
    query_p = subparsers.add_parser("query", help="Query the code knowledge graph")
    query_p.add_argument("query_str", help="Search string or symbol name")
    query_p.add_argument("-g", "--graph", default="graphify-out/graph.json", help="Path to graph.json")

    # Command: path
    path_p = subparsers.add_parser("path", help="Find path between two code components")
    path_p.add_argument("source", help="Source component or symbol name")
    path_p.add_argument("target", help="Target component or symbol name")
    path_p.add_argument("-g", "--graph", default="graphify-out/graph.json", help="Path to graph.json")

    # Command: explain
    explain_p = subparsers.add_parser("explain", help="Explain structural role and connections of a component")
    explain_p.add_argument("node", help="Symbol name or node ID")
    explain_p.add_argument("-g", "--graph", default="graphify-out/graph.json", help="Path to graph.json")

    return parser


def generate_graph(repo_dir: str, output_dir: str) -> int:
    """Executes full AST parsing and artifact generation."""
    print(f"[graphify] Scanning repository at: {os.path.abspath(repo_dir)}")
    parser = ASTCodeParser(repo_dir)
    ast_data = parser.parse_repository()

    total_symbols = (
        len(ast_data["modules"])
        + len(ast_data["classes"])
        + len(ast_data["functions"])
        + len(ast_data["docs"])
    )
    print(f"[graphify] Parsed {len(ast_data['modules'])} modules, {len(ast_data['classes'])} classes, {len(ast_data['functions'])} functions.")

    graph = CodeKnowledgeGraph(repo_dir)
    graph.build_from_ast(ast_data)
    print(f"[graphify] Knowledge graph built: {len(graph.nodes)} nodes, {len(graph.edges)} edges.")

    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "graph.json")
    html_path = os.path.join(output_dir, "graph.html")
    report_path = os.path.join(output_dir, "GRAPH_REPORT.md")

    export_graph_json(graph, json_path)
    print(f"[graphify] Generated machine-readable graph: {json_path}")

    export_graph_html(graph, html_path)
    print(f"[graphify] Generated interactive visualization: {html_path}")

    export_graph_report(graph, report_path)
    print(f"[graphify] Generated architectural report: {report_path}")

    print("[graphify] Code knowledge graph generation complete.")
    return 0


def load_graph_data(graph_path: str) -> Optional[CodeKnowledgeGraph]:
    if not os.path.isfile(graph_path):
        print(f"[graphify] Error: Graph file not found: {graph_path}. Run 'graphify .' first.", file=sys.stderr)
        return None

    try:
        with open(graph_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        graph = CodeKnowledgeGraph()
        for node in data.get("nodes", []):
            graph.nodes[node["id"]] = node
            if graph._nx_graph is not None:
                graph._nx_graph.add_node(node["id"], **node)
        for edge in data.get("edges", []):
            graph.edges.append(edge)
            graph._adj[edge["source"]].add(edge["target"])
            graph._rev_adj[edge["target"]].add(edge["source"])
            if graph._nx_graph is not None:
                graph._nx_graph.add_edge(edge["source"], edge["target"], relation=edge.get("relation", ""))
        return graph
    except Exception as e:
        print(f"[graphify] Failed to load graph: {e}", file=sys.stderr)
        return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    # Handle quick positional invocation like `graphify .`
    if len(argv) == 1 and argv[0] not in ("-h", "--help", "-v", "--version") and not argv[0].startswith("-"):
        return generate_graph(argv[0], "graphify-out")
    elif len(argv) == 0:
        return generate_graph(".", "graphify-out")

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in ("extract", None):
        target_dir = getattr(args, "directory", ".") or "."
        out_dir = getattr(args, "output_dir", "graphify-out") or "graphify-out"
        return generate_graph(target_dir, out_dir)

    elif args.command == "update":
        target_dir = getattr(args, "directory", ".") or "."
        out_dir = getattr(args, "output_dir", "graphify-out") or "graphify-out"
        print("[graphify] Updating and re-indexing knowledge graph...")
        return generate_graph(target_dir, out_dir)

    elif args.command == "query":
        graph = load_graph_data(args.graph)
        if not graph:
            return 1
        results = graph.query(args.query_str)
        print(f"\n[graphify] Query results for '{args.query_str}' ({len(results)} matches):\n")
        for i, res in enumerate(results, 1):
            print(f" {i}. [{res['type'].upper()}] {res['label']} ({res['id']})")
            if res.get("source_file"):
                print(f"    Location: {res['source_file']}:{res.get('line', 1)}")
            print(f"    Subsystem: {res['community']} | Centrality: {res['centrality']}")
            if res.get("docstring"):
                first_line = res["docstring"].splitlines()[0]
                print(f"    Doc: {first_line}")
            print()
        return 0

    elif args.command == "path":
        graph = load_graph_data(args.graph)
        if not graph:
            return 1
        path = graph.find_path(args.source, args.target)
        if path:
            print(f"\n[graphify] Path from '{args.source}' to '{args.target}' ({len(path)} hops):")
            print(" -> ".join(path) + "\n")
            return 0
        else:
            print(f"\n[graphify] No path found between '{args.source}' and '{args.target}'.\n")
            return 1

    elif args.command == "explain":
        graph = load_graph_data(args.graph)
        if not graph:
            return 1
        explanation = graph.explain_node(args.node)
        if not explanation:
            print(f"\n[graphify] Node '{args.node}' not found in knowledge graph.\n")
            return 1

        node = explanation["node"]
        print(f"\n[graphify] Component Breakdown: {node['label']} ({node['id']})")
        print(f" Type        : {node['type']}")
        print(f" Subsystem   : {node['community']}")
        print(f" Location    : {node.get('source_file', 'N/A')}:{node.get('line', 1)}")
        print(f" Centrality  : {node['centrality']} (In: {node['in_degree']}, Out: {node['out_degree']})")
        if node.get("docstring"):
            print(f" Docstring   :\n   {node['docstring']}")
        print("\n Inbound Callers & Dependents:")
        for c in explanation["callers_and_dependents"][:10]:
            print(f"   <- {c}")
        print("\n Outbound Dependencies & Callees:")
        for c in explanation["callees_and_dependencies"][:10]:
            print(f"   -> {c}")
        print()
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
