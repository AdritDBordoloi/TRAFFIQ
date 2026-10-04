"""
Knowledge Graph Engine for Graphify.
Constructs directed property graph from AST symbols, computes centrality metrics,
assigns modular architectural communities, and provides query / traversal APIs.
"""

from __future__ import annotations

import collections
import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import networkx as nx
except ImportError:
    nx = None


class CodeKnowledgeGraph:
    """Graph model managing codebase symbols, relations, and topological queries."""

    COMMUNITY_PALETTE = {
        "core_config": "#4A90E2",       # Blue
        "database": "#50E3C2",          # Teal
        "vision": "#F5A623",            # Orange
        "enforcement": "#D0021B",       # Red
        "cli_entrypoint": "#9013FE",    # Purple
        "testing_suite": "#7ED321",     # Green
        "scripts_tooling": "#B8E986",   # Light Green
        "documentation": "#9B9B9B",     # Grey
        "external_library": "#417505",  # Dark Green
        "general": "#F8E71C",           # Yellow
    }

    def __init__(self, root_dir: str = ".") -> None:
        self.root_dir = root_dir
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Dict[str, Any]] = []
        self._adj: Dict[str, Set[str]] = collections.defaultdict(set)
        self._rev_adj: Dict[str, Set[str]] = collections.defaultdict(set)
        if nx is not None:
            self._nx_graph = nx.DiGraph()
        else:
            self._nx_graph = None

    def build_from_ast(self, ast_data: Dict[str, Any]) -> None:
        """Populates graph from ASTCodeParser output."""
        self.root_dir = ast_data.get("root_dir", self.root_dir)

        # 1. Modules
        for mod in ast_data.get("modules", []):
            community = self._classify_community(mod["id"])
            self.add_node(
                node_id=mod["id"],
                label=mod["name"],
                node_type="module",
                source_file=mod["file"],
                line=1,
                docstring=mod.get("docstring", ""),
                community=community,
                metadata={"lines": mod.get("lines", 0), "size": mod.get("size", 0)},
            )

        # 2. Classes
        for cls in ast_data.get("classes", []):
            community = self._classify_community(cls["id"])
            self.add_node(
                node_id=cls["id"],
                label=cls["name"],
                node_type="class",
                source_file=cls["file"],
                line=cls["line"],
                docstring=cls.get("docstring", ""),
                community=community,
                metadata={"bases": cls.get("bases", [])},
            )
            # Edge: Module contains Class
            self.add_edge(cls["module"], cls["id"], relation="contains", weight=1.0)
            # Edge: Inherits
            for base in cls.get("bases", []):
                self.add_edge(cls["id"], base, relation="inherits", weight=1.5)

        # 3. Functions & Methods
        for fn in ast_data.get("functions", []):
            community = self._classify_community(fn["id"])
            node_type = "method" if fn.get("is_method") else "function"
            self.add_node(
                node_id=fn["id"],
                label=fn["name"],
                node_type=node_type,
                source_file=fn["file"],
                line=fn["line"],
                docstring=fn.get("docstring", ""),
                community=community,
                metadata={"args": fn.get("args", []), "class_name": fn.get("class_name")},
            )
            # Edge: Parent contains Function/Method
            self.add_edge(fn["parent_id"], fn["id"], relation="contains", weight=1.0)

        # 4. Imports
        for imp in ast_data.get("imports", []):
            source = imp["source"]
            target = imp["target"]
            self.add_edge(source, target, relation="imports", weight=1.0)

        # 5. Calls
        for call in ast_data.get("calls", []):
            caller = call["caller"]
            callee = call["callee"]
            self.add_edge(caller, callee, relation="calls", weight=1.0)

        # 6. Documentation & Configs
        for doc in ast_data.get("docs", []):
            self.add_node(
                node_id=doc["id"],
                label=doc["name"],
                node_type=doc["type"],
                source_file=doc["file"],
                line=1,
                docstring=f"Project {doc['type']} file",
                community="documentation",
                metadata={"size": doc.get("size", 0)},
            )
            self.add_edge(doc["id"], "traffiq", relation="documents", weight=0.5)

        # Calculate metrics
        self._compute_graph_metrics()

    def add_node(
        self,
        node_id: str,
        label: str,
        node_type: str,
        source_file: str = "",
        line: int = 1,
        docstring: str = "",
        community: str = "general",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if node_id not in self.nodes:
            self.nodes[node_id] = {
                "id": node_id,
                "label": label,
                "type": node_type,
                "source_file": source_file,
                "line": line,
                "docstring": docstring,
                "community": community,
                "in_degree": 0,
                "out_degree": 0,
                "centrality": 0.0,
                "metadata": metadata or {},
            }
        else:
            # Update existing node if new info has higher fidelity
            existing = self.nodes[node_id]
            if not existing.get("docstring") and docstring:
                existing["docstring"] = docstring
            if not existing.get("source_file") and source_file:
                existing["source_file"] = source_file
                existing["line"] = line

        if self._nx_graph is not None:
            self._nx_graph.add_node(
                node_id,
                label=label,
                type=node_type,
                community=community,
                file=source_file,
            )

    def add_edge(self, source: str, target: str, relation: str = "references", weight: float = 1.0) -> None:
        if not source or not target or source == target:
            return

        # Ensure target node exists
        if target not in self.nodes:
            target_comm = self._classify_community(target)
            self.add_node(
                node_id=target,
                label=target.split(".")[-1],
                node_type="external" if not target.startswith("traffiq") and not target.startswith("tests") else "symbol",
                community=target_comm,
            )

        if source not in self.nodes:
            source_comm = self._classify_community(source)
            self.add_node(
                node_id=source,
                label=source.split(".")[-1],
                node_type="symbol",
                community=source_comm,
            )

        edge = {
            "source": source,
            "target": target,
            "relation": relation,
            "weight": weight,
        }
        self.edges.append(edge)
        self._adj[source].add(target)
        self._rev_adj[target].add(source)

        if self._nx_graph is not None:
            self._nx_graph.add_edge(source, target, relation=relation, weight=weight)

    def _classify_community(self, symbol_path: str) -> str:
        s = symbol_path.lower()
        if s.startswith("traffiq.config"):
            return "core_config"
        elif s.startswith("traffiq.database"):
            return "database"
        elif s.startswith("traffiq.vision"):
            return "vision"
        elif s.startswith("traffiq.enforcement"):
            return "enforcement"
        elif s.startswith("traffiq.cli") or s == "main" or s.startswith("main."):
            return "cli_entrypoint"
        elif s.startswith("tests"):
            return "testing_suite"
        elif s.startswith("scripts"):
            return "scripts_tooling"
        elif s.startswith("doc:"):
            return "documentation"
        elif any(s.startswith(lib) for lib in ("torch", "cv2", "ultralytics", "easyocr", "sqlite3", "numpy")):
            return "external_library"
        return "general"

    def _compute_graph_metrics(self) -> None:
        n_count = len(self.nodes)
        for nid, data in self.nodes.items():
            out_deg = len(self._adj.get(nid, set()))
            in_deg = len(self._rev_adj.get(nid, set()))
            data["out_degree"] = out_deg
            data["in_degree"] = in_deg
            total_deg = in_deg + out_deg
            data["centrality"] = round(total_deg / max(1, n_count - 1), 4)

        if self._nx_graph is not None and len(self._nx_graph) > 0:
            try:
                centralities = nx.degree_centrality(self._nx_graph)
                for nid, cent in centralities.items():
                    if nid in self.nodes:
                        self.nodes[nid]["centrality"] = round(cent, 4)
            except Exception:
                pass

    def get_god_nodes(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """Returns top connected nodes (God nodes) in the codebase."""
        sorted_nodes = sorted(
            [n for n in self.nodes.values() if n["type"] != "external"],
            key=lambda x: (x["in_degree"] + x["out_degree"], x["in_degree"]),
            reverse=True,
        )
        return sorted_nodes[:top_n]

    def find_path(self, source: str, target: str) -> Optional[List[str]]:
        """Finds shortest path between two nodes using BFS."""
        if source not in self.nodes or target not in self.nodes:
            # Try fuzzy match
            src_matched = self._match_single_node(source)
            tgt_matched = self._match_single_node(target)
            if not src_matched or not tgt_matched:
                return None
            source = src_matched
            target = tgt_matched

        if self._nx_graph is not None:
            try:
                return nx.shortest_path(self._nx_graph, source=source, target=target)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass

        # Manual BFS fallback
        queue = collections.deque([[source]])
        visited = {source}

        while queue:
            path = queue.popleft()
            curr = path[-1]
            if curr == target:
                return path

            for neighbor in self._adj.get(curr, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])

        return None

    def explain_node(self, query: str) -> Optional[Dict[str, Any]]:
        """Returns deep structural context, incoming callers, and outgoing dependencies."""
        node_id = self._match_single_node(query)
        if not node_id:
            return None

        node = self.nodes[node_id]
        incoming = list(self._rev_adj.get(node_id, set()))
        outgoing = list(self._adj.get(node_id, set()))

        return {
            "node": node,
            "callers_and_dependents": incoming,
            "callees_and_dependencies": outgoing,
            "community": node["community"],
            "centrality_score": node["centrality"],
        }

    def query(self, search_term: str) -> List[Dict[str, Any]]:
        """Searches for nodes matching term in label, docstring, or ID."""
        term = search_term.lower()
        matches = []
        for n in self.nodes.values():
            score = 0
            if term == n["label"].lower():
                score += 10
            elif term in n["label"].lower():
                score += 5
            elif term in n["id"].lower():
                score += 3
            elif term in n.get("docstring", "").lower():
                score += 1

            if score > 0:
                matches.append((score, n))

        matches.sort(key=lambda x: (x[0], x[1]["centrality"]), reverse=True)
        return [item[1] for item in matches[:20]]

    def _match_single_node(self, term: str) -> Optional[str]:
        if term in self.nodes:
            return term
        term_lower = term.lower()
        # Exact label match
        for nid, data in self.nodes.items():
            if data["label"].lower() == term_lower:
                return nid
        # Substring in ID
        for nid in self.nodes:
            if term_lower in nid.lower():
                return nid
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes knowledge graph into JSON-compatible dictionary."""
        community_counts = collections.Counter(n["community"] for n in self.nodes.values())
        return {
            "metadata": {
                "generator": "graphifyy",
                "version": "0.2.0",
                "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "project_root": self.root_dir,
                "total_nodes": len(self.nodes),
                "total_edges": len(self.edges),
                "communities": dict(community_counts),
            },
            "nodes": list(self.nodes.values()),
            "edges": self.edges,
            "god_nodes": self.get_god_nodes(5),
        }
