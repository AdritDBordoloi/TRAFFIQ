"""Tier 4: CLI Process Execution & Entrypoint Integration Tests.
Verifies `python main.py --help` exits with code 0 and displays all operational modes,
verifies invalid CLI options are rejected with non-zero exit codes, and
verifies headless diagnostics execution.
"""

import os
import sys
import subprocess
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class TestCLIExecution(unittest.TestCase):
    """Tests command-line interface execution via subprocess."""

    def test_cli_help_displays_operational_modes(self):
        """Verifies python main.py --help exits code 0 and documents all 4 operational modes."""
        cmd = [sys.executable, "main.py", "--help"]
        res = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(
            res.returncode, 0,
            f"main.py --help exited with non-zero code {res.returncode}.\nStderr: {res.stderr}\nStdout: {res.stdout}",
        )

        stdout = res.stdout.lower()
        required_modes = ["diagnostics", "detection", "tracking", "enforcement"]
        for mode in required_modes:
            self.assertIn(
                mode, stdout,
                f"Operational mode '{mode}' not documented in main.py --help output",
            )

        self.assertIn("--no-gui", stdout, "--no-gui flag not documented in main.py --help")

    def test_cli_invalid_mode_rejected(self):
        """Verifies unrecognized mode choices exit with non-zero return code."""
        cmd = [sys.executable, "main.py", "--mode", "nonexistent_mode_xyz"]
        res = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertNotEqual(
            res.returncode, 0,
            "main.py should exit with non-zero code when provided an invalid mode choice",
        )

    def test_cli_diagnostics_headless_mode(self):
        """Verifies diagnostics mode runs cleanly with --no-gui and --max-frames 1."""
        # First check if main.py supports --mode diagnostics
        help_res = subprocess.run(
            [sys.executable, "main.py", "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if "--mode" not in help_res.stdout:
            self.skipTest("main.py CLI dispatcher not yet implemented (scheduled for M3)")

        cmd = [
            sys.executable, "main.py",
            "--mode", "diagnostics",
            "--no-gui",
            "--max-frames", "1",
        ]
        res = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(
            res.returncode, 0,
            f"Diagnostics mode failed with returncode {res.returncode}.\nStderr: {res.stderr}\nStdout: {res.stdout}",
        )


if __name__ == "__main__":
    unittest.main()
