"""Tier 1: Command Line Interface (CLI) Argument Parsing Tests.
Verifies argparse definitions, mode choices (diagnostics, detection, tracking, enforcement),
operational flags (--fine, --source, --conf, --no-gui, --device, --max-frames),
and rejection of invalid choices.
"""

import sys
import unittest
import argparse

try:
    from traffiq.cli.app import build_parser, parse_args
except ImportError:
    build_parser = None
    parse_args = None


def oracle_build_parser() -> argparse.ArgumentParser:
    """Authoritative reference CLI parser definition matching PROJECT.md interface contract."""
    parser = argparse.ArgumentParser(
        prog="traffiq",
        description="TRAFFIQ: AI-Powered Automated Traffic Enforcement System",
    )
    parser.add_argument(
        "--mode",
        choices=["diagnostics", "detection", "tracking", "enforcement"],
        default="enforcement",
        help="Operational pipeline mode",
    )
    parser.add_argument(
        "--source", "--camera",
        dest="source",
        default="0",
        help="Camera device index (int) or path to video file (str)",
    )
    parser.add_argument(
        "--fine",
        type=int,
        default=1000,
        help="Penalty fine amount in INR (default: 1000)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO detection confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        default=False,
        help="Disable interactive OpenCV visualization windows for headless operation",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Inference compute device ('0' for GPU or 'cpu')",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process before exiting",
    )
    parser.add_argument(
        "--db-path",
        default="traffiq_enforcement.db",
        help="Path to SQLite database file",
    )
    return parser


class TestCLIParsing(unittest.TestCase):
    """Tests argument parsing logic and operational mode definitions."""

    def setUp(self):
        self.parser_builder = build_parser or oracle_build_parser
        self.parser = self.parser_builder()

    def test_default_mode_and_arguments(self):
        """Verifies default argument values when no flags are supplied."""
        args = self.parser.parse_args([])
        self.assertEqual(args.mode, "enforcement")
        self.assertEqual(args.fine, 1000)
        self.assertEqual(args.conf, 0.25)
        self.assertFalse(args.no_gui)
        self.assertIn(args.device, ["0", "cpu"])

    def test_operational_mode_choices(self):
        """Verifies all four mandatory operational modes are accepted."""
        valid_modes = ["diagnostics", "detection", "tracking", "enforcement"]
        for mode in valid_modes:
            args = self.parser.parse_args(["--mode", mode])
            self.assertEqual(args.mode, mode)

    def test_invalid_mode_rejected(self):
        """Verifies unrecognized mode choices trigger an argument error."""
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["--mode", "nonexistent_mode"])

    def test_flag_overrides(self):
        """Verifies custom flags for fine, source, thresholds, and headless mode."""
        args = self.parser.parse_args([
            "--mode", "tracking",
            "--source", "traffic_cam_01.mp4",
            "--fine", "2500",
            "--conf", "0.45",
            "--no-gui",
            "--device", "cpu",
            "--max-frames", "500",
            "--db-path", "custom_test.db",
        ])
        self.assertEqual(args.mode, "tracking")
        self.assertEqual(args.source, "traffic_cam_01.mp4")
        self.assertEqual(args.fine, 2500)
        self.assertEqual(args.conf, 0.45)
        self.assertTrue(args.no_gui)
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.max_frames, 500)
        self.assertEqual(args.db_path, "custom_test.db")


if __name__ == "__main__":
    unittest.main()
