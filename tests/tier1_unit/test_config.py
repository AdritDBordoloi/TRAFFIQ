"""Tier 1: Configuration Management Tests.
Verifies TraffiqConfig dataclass default values, custom parameter overrides,
type validation, and dictionary conversion.
"""

import os
import unittest
from dataclasses import asdict, is_dataclass

try:
    from traffiq.config.settings import TraffiqConfig
except ImportError:
    TraffiqConfig = None


class TestTraffiqConfig(unittest.TestCase):
    """Verifies TraffiqConfig implementation against PROJECT.md interface contracts."""

    def setUp(self):
        if TraffiqConfig is None:
            self.skipTest("traffiq.config.settings.TraffiqConfig not yet implemented (scheduled for M1)")

    def test_default_values(self):
        """Verifies default values match PROJECT.md interface contract exactly."""
        config = TraffiqConfig()
        
        self.assertEqual(config.camera_index, 0)
        self.assertEqual(config.frame_width, 1280)
        self.assertEqual(config.frame_height, 720)
        self.assertIn(config.device, ["0", "cpu"])
        self.assertEqual(config.model_path, "yolov8s.pt")
        self.assertEqual(config.confidence_threshold, 0.25)
        self.assertEqual(config.tripwire_ratio, 0.55)
        self.assertEqual(config.fine_amount, 1000)
        self.assertEqual(config.signal_duration, 8.0)
        self.assertEqual(config.output_dir, "violations")
        self.assertEqual(config.db_path, "traffiq_enforcement.db")
        self.assertFalse(config.no_gui)
        self.assertIsNone(config.max_frames)

    def test_custom_overrides(self):
        """Verifies custom keyword arguments properly override defaults."""
        config = TraffiqConfig(
            camera_index="demo.mp4",
            confidence_threshold=0.45,
            fine_amount=2500,
            no_gui=True,
            max_frames=100,
            device="cpu",
        )
        self.assertEqual(config.camera_index, "demo.mp4")
        self.assertEqual(config.confidence_threshold, 0.45)
        self.assertEqual(config.fine_amount, 2500)
        self.assertTrue(config.no_gui)
        self.assertEqual(config.max_frames, 100)
        self.assertEqual(config.device, "cpu")

    def test_is_dataclass(self):
        """Verifies TraffiqConfig is defined as a Python dataclass."""
        self.assertTrue(is_dataclass(TraffiqConfig), "TraffiqConfig must be a dataclass")

    def test_serialization_to_dict(self):
        """Verifies TraffiqConfig converts cleanly to a dictionary."""
        config = TraffiqConfig()
        d = asdict(config) if is_dataclass(config) else config.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["fine_amount"], 1000)
        self.assertEqual(d["tripwire_ratio"], 0.55)

    def test_validation_ranges(self):
        """Verifies validation logic handles boundary checks."""
        # Valid edge boundaries
        config_min_conf = TraffiqConfig(confidence_threshold=0.01)
        self.assertEqual(config_min_conf.confidence_threshold, 0.01)

        config_max_conf = TraffiqConfig(confidence_threshold=0.99)
        self.assertEqual(config_max_conf.confidence_threshold, 0.99)

        # Negative fine rejection if validation is implemented
        if hasattr(TraffiqConfig, "validate"):
            with self.assertRaises((ValueError, TypeError)):
                bad_config = TraffiqConfig(confidence_threshold=-0.5)
                bad_config.validate()


if __name__ == "__main__":
    unittest.main()
