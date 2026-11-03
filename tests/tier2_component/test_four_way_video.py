"""Tier 2: Component Tests for 4-Way Video Multi-Stream & Quadrant Processor.
Verifies approach zone partitioning, quadrant/crossroad centroid mapping,
annotation HUD rendering, and video capture lifecycle management.
"""

import unittest
import numpy as np

try:
    from traffiq.enforcement.adaptive_signal import AdaptiveIntersectionController, Approach, SignalColor
    from traffiq.vision.four_way import FourWayVideoProcessor
except (ImportError, ModuleNotFoundError, AttributeError):
    AdaptiveIntersectionController = None
    Approach = None
    SignalColor = None
    FourWayVideoProcessor = None


class TestFourWayVideoProcessor(unittest.TestCase):
    """Test suite for 4-way intersection video input processing."""

    def setUp(self):
        if FourWayVideoProcessor is None or AdaptiveIntersectionController is None:
            self.skipTest("four_way or adaptive_signal not yet implemented")
        self.processor = FourWayVideoProcessor(mode="quadrant", device="cpu")
        self.controller = AdaptiveIntersectionController()

    def tearDown(self):
        if hasattr(self, "processor") and self.processor is not None:
            self.processor.close()

    def test_quadrant_centroid_assignment(self):
        """In quadrant mode (2x2 grid), centroids map to North, East, West, South."""
        w, h = 1280, 720
        # Top-Left -> NORTH
        self.assertEqual(self.processor._assign_approach(200, 150, w, h), Approach.NORTH.value)
        # Top-Right -> EAST
        self.assertEqual(self.processor._assign_approach(900, 150, w, h), Approach.EAST.value)
        # Bottom-Left -> WEST
        self.assertEqual(self.processor._assign_approach(200, 500, w, h), Approach.WEST.value)
        # Bottom-Right -> SOUTH
        self.assertEqual(self.processor._assign_approach(900, 500, w, h), Approach.SOUTH.value)

    def test_crossroad_centroid_assignment(self):
        """In crossroad mode (overhead view), angles from center determine approach."""
        proc = FourWayVideoProcessor(mode="crossroad", device="cpu")
        w, h = 1280, 720
        cx, cy = 640, 360

        # High above center -> NORTH
        self.assertEqual(proc._assign_approach(cx, 100, w, h), Approach.NORTH.value)
        # Far right of center -> EAST
        self.assertEqual(proc._assign_approach(1100, cy, w, h), Approach.EAST.value)
        # Far below center -> SOUTH
        self.assertEqual(proc._assign_approach(cx, 650, w, h), Approach.SOUTH.value)
        # Far left of center -> WEST
        self.assertEqual(proc._assign_approach(100, cy, w, h), Approach.WEST.value)
        proc.close()

    def test_synthetic_frame_generation(self):
        """When no video is supplied, returns valid 1280x720 3-channel frame."""
        has_src, frame = self.processor.read_raw_frame()
        self.assertFalse(has_src)
        self.assertIsInstance(frame, np.ndarray)
        self.assertEqual(frame.shape, (720, 1280, 3))
        self.assertEqual(frame.dtype, np.uint8)

    def test_process_and_annotate(self):
        """Verifies HUD drawing and count extraction without crashing."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        annotated, counts, emergencies = self.processor.process_and_annotate(frame, self.controller)

        self.assertEqual(annotated.shape, (720, 1280, 3))
        self.assertIn("NORTH", counts)
        self.assertIn("SOUTH", counts)
        self.assertIn("EAST", counts)
        self.assertIn("WEST", counts)

    def test_set_sources_invalid_handled_gracefully(self):
        """Passing non-existent video path returns False and does not crash."""
        res = self.processor.set_sources(source_4way="non_existent_clip_123.mp4")
        self.assertFalse(res)


if __name__ == "__main__":
    unittest.main()
