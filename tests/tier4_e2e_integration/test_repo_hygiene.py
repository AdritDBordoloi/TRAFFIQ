"""Tier 4: Repository Hygiene & Git Porcelain State Tests.
Verifies the existence and non-emptiness of standard open-source documentation files
(LICENSE, CONTRIBUTING.md, README.md, requirements.txt, .gitignore) and
verifies git status reports 0 exposed *.pt weights, *.db databases, .venv, or violations/ images.
"""

import os
import subprocess
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class TestRepoHygiene(unittest.TestCase):
    """Tests open-source documentation completeness and git index cleanliness."""

    def test_required_documentation_files_exist(self):
        """Verifies presence of LICENSE, CONTRIBUTING.md, README.md, requirements.txt, and .gitignore."""
        required_files = [
            "LICENSE",
            "CONTRIBUTING.md",
            "README.md",
            "requirements.txt",
            ".gitignore",
        ]
        missing = []
        empty = []

        for fname in required_files:
            fpath = os.path.join(PROJECT_ROOT, fname)
            if not os.path.isfile(fpath):
                missing.append(fname)
            elif os.path.getsize(fpath) == 0:
                empty.append(fname)

        self.assertFalse(missing, f"Missing required repository documentation files: {missing} (scheduled for M5)")
        self.assertFalse(empty, f"Required repository files are empty: {empty}")

    def test_license_content(self):
        """Verifies LICENSE file contains MIT license terms."""
        license_path = os.path.join(PROJECT_ROOT, "LICENSE")
        if not os.path.exists(license_path):
            self.skipTest("LICENSE file not yet created (scheduled for M5)")

        with open(license_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("MIT License", content)
        self.assertIn("2026", content)

    def test_gitignore_covers_sensitive_and_binary_patterns(self):
        """Verifies .gitignore contains rules for models, databases, environments, and test caches."""
        gitignore_path = os.path.join(PROJECT_ROOT, ".gitignore")
        self.assertTrue(os.path.isfile(gitignore_path), ".gitignore is missing")

        with open(gitignore_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines()]

        mandatory_patterns = [
            ".venv",
            "*.pt",
            "*.db",
            "violations",
        ]
        for pat in mandatory_patterns:
            matched = any(pat in line for line in lines)
            self.assertTrue(matched, f".gitignore is missing mandatory pattern rule: {pat}")

    def test_git_status_reports_zero_exposed_binaries(self):
        """Verifies git status --porcelain reports 0 exposed *.pt weights, *.db databases, .venv, or violation captures."""
        res = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0, f"git status failed: {res.stderr}")

        forbidden_extensions = (".pt", ".db", ".sqlite3", ".pyc")
        lines = res.stdout.splitlines()

        for line in lines:
            # Format: '?? path/to/file' or ' M path/to/file'
            parts = line.strip().split(maxsplit=1)
            if len(parts) < 2:
                continue
            filepath = parts[1]

            for ext in forbidden_extensions:
                self.assertFalse(
                    filepath.endswith(ext),
                    f"Forbidden binary artifact exposed in git status: {filepath}",
                )

            self.assertNotIn(
                ".venv", filepath,
                f"Virtual environment directory exposed in git status: {filepath}",
            )
            self.assertNotIn(
                "violations/", filepath,
                f"Violation evidence capture exposed in git status: {filepath}",
            )


if __name__ == "__main__":
    unittest.main()
