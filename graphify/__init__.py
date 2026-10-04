"""
Graphify (graphifyy) — AST-Driven Codebase Knowledge Graph Generator & Explorer.
Transforms codebases, schemas, configs, and documentation into queryable,
force-directed knowledge graphs without external API dependencies.
"""

from __future__ import annotations

__version__ = "0.2.0"
__author__ = "Graphify Labs"
__license__ = "MIT"

from .ast_parser import ASTCodeParser
from .graph import CodeKnowledgeGraph
from .exporters import export_graph_json, export_graph_html, export_graph_report

__all__ = [
    "__version__",
    "ASTCodeParser",
    "CodeKnowledgeGraph",
    "export_graph_json",
    "export_graph_html",
    "export_graph_report",
]
