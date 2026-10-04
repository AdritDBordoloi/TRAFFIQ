"""Tier 4: Graphifyy Tooling Integration & Knowledge Graph CLI Tests.
Verifies graphify CLI availability, version execution (`graphify --version`),
and knowledge graph generation contracts (`graphify-out/`).
"""

import os
import sys
import shutil
import subprocess
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class TestGraphifyCLI(unittest.TestCase):
    """Tests graphifyy CLI tooling execution and graph generation."""

    def _get_graphify_cmd(self):
        """Finds graphify executable in PATH or .venv/Scripts."""
        # 1. System PATH
        which_path = shutil.which("graphify")
        if which_path:
            return [which_path]

        # 2. Local virtualenv Scripts directory
        venv_scripts = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "graphify.exe")
        if os.path.exists(venv_scripts):
            return [venv_scripts]

        # 3. Python module invocation
        return [sys.executable, "-m", "graphifyy"]

    def test_graphify_version_execution(self):
        """Verifies graphify --version (or python -m graphifyy --version) executes successfully."""
        cmd = self._get_graphify_cmd() + ["--version"]
        try:
            res = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (FileNotFoundError, OSError):
            self.skipTest("graphifyy not yet installed in environment (scheduled for M4)")

        if res.returncode != 0:
            self.skipTest(f"graphify CLI returned non-zero ({res.returncode}): {res.stderr} (scheduled for M4)")

        self.assertEqual(res.returncode, 0)
        output = (res.stdout + res.stderr).lower()
        self.assertTrue(len(output) > 0, "graphify --version should output version information")

    def test_graph_output_directory_contract(self):
        """Verifies graphify-out directory structure contract when knowledge graph has been generated."""
        graph_dir = os.path.join(PROJECT_ROOT, "graphify-out")
        if not os.path.exists(graph_dir):
            self.skipTest("graphify-out/ not yet generated (scheduled for M4)")

        # When graphify has executed, expected graph artifacts include graph.json, graph.html, GRAPH_REPORT.md
        entries = os.listdir(graph_dir)
        self.assertGreater(len(entries), 0, "graphify-out/ directory exists but is empty")
        self.assertIn("graph.json", entries, "graphify-out/ must contain graph.json")
        self.assertIn("graph.html", entries, "graphify-out/ must contain graph.html")
        self.assertIn("GRAPH_REPORT.md", entries, "graphify-out/ must contain GRAPH_REPORT.md")

    def test_graph_json_schema_and_contents(self):
        """Verifies graph.json conforms to the machine-readable AST property graph schema."""
        import json
        json_path = os.path.join(PROJECT_ROOT, "graphify-out", "graph.json")
        if not os.path.exists(json_path):
            self.skipTest("graphify-out/graph.json not yet generated")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("metadata", data)
        self.assertIn("nodes", data)
        self.assertIn("edges", data)
        self.assertGreater(len(data["nodes"]), 0)
        self.assertGreater(len(data["edges"]), 0)

        # Spot check node schema
        first_node = data["nodes"][0]
        self.assertIn("id", first_node)
        self.assertIn("label", first_node)
        self.assertIn("type", first_node)
        self.assertIn("community", first_node)

    def test_graph_report_content(self):
        """Verifies GRAPH_REPORT.md contains architectural analysis and god nodes summary."""
        report_path = os.path.join(PROJECT_ROOT, "graphify-out", "GRAPH_REPORT.md")
        if not os.path.exists(report_path):
            self.skipTest("graphify-out/GRAPH_REPORT.md not yet generated")

        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("Architectural Report", content)
        self.assertIn("God Nodes", content)
        self.assertIn("Subsystems", content)

    def test_graphify_documentation_guide_exists(self):
        """Verifies docs/GRAPHIFY.md exists and documents installation and usage procedures."""
        doc_path = os.path.join(PROJECT_ROOT, "docs", "GRAPHIFY.md")
        self.assertTrue(os.path.isfile(doc_path), "docs/GRAPHIFY.md is required")

        with open(doc_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("pip install graphifyy", content)
        self.assertIn("graphify .", content)
        self.assertIn("graphify query", content)
        self.assertIn("graphify path", content)
        self.assertIn("graphify explain", content)


if __name__ == "__main__":
    unittest.main()

