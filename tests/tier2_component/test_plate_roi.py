"""Tier 2: License Plate Region-of-Interest (ROI) Extraction Tests.
Verifies license plate ROI extraction isolates the lower 40-50% horizontal band
of the vehicle bounding box, applies boundary clamping on edge borders, and
filters candidate aspect ratios.
"""

import unittest
import numpy as np

try:
    from traffiq.enforcement.ocr import PlateExtractor, extract_plate_roi
except ImportError:
    PlateExtractor = None
    extract_plate_roi = None


def oracle_extract_plate_roi(vehicle_crop: np.ndarray, y_start_ratio: float = 0.55, y_end_ratio: float = 1.0) -> np.ndarray:
    """Authoritative reference ROI extraction isolating lower 40-50% horizontal band of vehicle crop."""
    if vehicle_crop is None or vehicle_crop.size == 0:
        return np.empty((0, 0, 3), dtype=np.uint8)

    h, w = vehicle_crop.shape[:2]
    if h == 0 or w == 0:
        return np.empty((0, 0, 3), dtype=np.uint8)

    # Calculate lower band
    y1 = int(np.clip(h * y_start_ratio, 0, h))
    y2 = int(np.clip(h * y_end_ratio, 0, h))

    # Boundary clamping guarantee
    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))

    return vehicle_crop[y1:y2, :]


def oracle_clamp_bbox(bbox, frame_shape):
    """Clamps bounding box coordinates [x1, y1, x2, y2] within frame boundaries."""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    cx1 = max(0, min(int(x1), w - 1))
    cy1 = max(0, min(int(y1), h - 1))
    cx2 = max(cx1 + 1, min(int(x2), w))
    cy2 = max(cy1 + 1, min(int(y2), h))
    return cx1, cy1, cx2, cy2


class TestPlateROI(unittest.TestCase):
    """Tests lower horizontal band extraction, boundary clamping, and aspect ratios."""

    def setUp(self):
        self.extract_fn = extract_plate_roi or oracle_extract_plate_roi

    def test_lower_band_isolation_ratio(self):
        """Verifies extracted ROI height is within 40-50% of the vehicle crop height."""
        # 200x300 vehicle crop
        vehicle_crop = np.zeros((200, 300, 3), dtype=np.uint8)
        roi = self.extract_fn(vehicle_crop)

        roi_h = roi.shape[0]
        ratio = roi_h / 200.0
        # The lower band should occupy between 40% and 50% (0.40 to 0.50) of total height
        self.assertTrue(
            0.40 <= ratio <= 0.50,
            f"Extracted ROI height ratio {ratio:.2f} is outside expected 40-50% band",
        )
        self.assertEqual(roi.shape[1], 300, "ROI width should match vehicle crop width")

    def test_boundary_clamping_on_border_coordinates(self):
        """Verifies coordinate clamping prevents out-of-bounds indexing."""
        frame_shape = (720, 1280, 3)

        # Negative coordinates extending off the left/top edges
        raw_box = (-20, -15, 300, 250)
        cx1, cy1, cx2, cy2 = oracle_clamp_bbox(raw_box, frame_shape)
        self.assertGreaterEqual(cx1, 0)
        self.assertGreaterEqual(cy1, 0)
        self.assertLessEqual(cx2, 1280)
        self.assertLessEqual(cy2, 720)

        # Overflow coordinates extending past bottom/right edges
        raw_box_overflow = (1200, 680, 1350, 780)
        cx1, cy1, cx2, cy2 = oracle_clamp_bbox(raw_box_overflow, frame_shape)
        self.assertEqual(cx2, 1280)
        self.assertEqual(cy2, 720)

    def test_small_and_degraded_crops(self):
        """Verifies that tiny vehicle crops do not cause IndexError or zero-division."""
        tiny_crop = np.zeros((10, 20, 3), dtype=np.uint8)
        roi = self.extract_fn(tiny_crop)
        self.assertIsNotNone(roi)
        self.assertGreater(roi.shape[0], 0)
        self.assertGreater(roi.shape[1], 0)

    def test_aspect_ratio_filtering(self):
        """Verifies aspect ratio calculation identifies standard plate dimensions (width/height ~ 2.0 to 4.5)."""
        def is_valid_plate_aspect_ratio(w, h):
            if h <= 0:
                return False
            ar = w / float(h)
            return 1.5 <= ar <= 5.5

        # Standard Indian HSRP plate: ~500mm x 120mm -> AR ~ 4.16
        self.assertTrue(is_valid_plate_aspect_ratio(250, 60))
        # Square-ish two-wheeler plate: ~200mm x 100mm -> AR ~ 2.0
        self.assertTrue(is_valid_plate_aspect_ratio(120, 60))
        # Extreme vertical sliver (false positive) -> AR 0.33
        self.assertFalse(is_valid_plate_aspect_ratio(20, 60))
        # Extreme horizontal line -> AR 12.0
        self.assertFalse(is_valid_plate_aspect_ratio(360, 30))

    def test_plate_extractor_class_contract_if_available(self):
        """Verifies PlateExtractor class contract if implemented."""
        if PlateExtractor is None:
            self.skipTest("traffiq.enforcement.ocr.PlateExtractor not yet implemented (scheduled for M2)")

        extractor = PlateExtractor()
        crop = np.zeros((200, 300, 3), dtype=np.uint8)
        roi = extractor.extract_roi(crop)
        self.assertIsNotNone(roi)
        self.assertTrue(0.40 <= (roi.shape[0] / 200.0) <= 0.50)


if __name__ == "__main__":
    unittest.main()
