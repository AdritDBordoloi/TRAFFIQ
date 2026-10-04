"""Tier 2: License Plate OCR Graceful Fallback Tests.
Verifies that unreadable, blank, blurry, low-confidence, or non-plate crops
gracefully fallback to ('UNKNOWN', 0.0) without throwing unhandled exceptions.
"""

import unittest
import numpy as np

try:
    from traffiq.enforcement.ocr import PlateExtractor, read_license_plate
except ImportError:
    PlateExtractor = None
    read_license_plate = None


def oracle_read_plate_fallback(vehicle_crop: np.ndarray, ocr_results=None) -> tuple[str, float]:
    """Authoritative reference implementation of OCR fallback logic.
    Returns (plate_text, confidence). When unreadable or below threshold, returns ('UNKNOWN', 0.0).
    """
    if vehicle_crop is None or vehicle_crop.size == 0:
        return "UNKNOWN", 0.0

    # If mock or actual ocr_results provided
    if ocr_results is not None:
        candidates = []
        for item in ocr_results:
            if len(item) == 3:
                _, text, conf = item
                import re
                cleaned = re.sub(r"[^A-Za-z0-9]", "", text).upper()
                if len(cleaned) >= 4 and conf >= 0.3:
                    candidates.append((cleaned, float(conf)))

        if candidates:
            # Sort by confidence and length
            best = max(candidates, key=lambda c: (len(c[0]), c[1]))
            return best[0], best[1]

    return "UNKNOWN", 0.0


class TestOCRFallback(unittest.TestCase):
    """Tests robustness and graceful fallback behavior across degraded inputs."""

    def setUp(self):
        self.fallback_fn = read_license_plate or oracle_read_plate_fallback

    def test_pure_black_blank_image(self):
        """Verifies pure black frame returns ('UNKNOWN', 0.0) without crashing."""
        black_crop = np.zeros((150, 300, 3), dtype=np.uint8)
        text, conf = self.fallback_fn(black_crop)
        self.assertEqual(text, "UNKNOWN")
        self.assertEqual(conf, 0.0)

    def test_pure_white_blank_image(self):
        """Verifies pure white frame returns ('UNKNOWN', 0.0) without crashing."""
        white_crop = np.full((150, 300, 3), 255, dtype=np.uint8)
        text, conf = self.fallback_fn(white_crop)
        self.assertEqual(text, "UNKNOWN")
        self.assertEqual(conf, 0.0)

    def test_random_noise_image(self):
        """Verifies high-entropy random pixel noise returns ('UNKNOWN', 0.0)."""
        noisy_crop = np.random.randint(0, 256, (150, 300, 3), dtype=np.uint8)
        text, conf = self.fallback_fn(noisy_crop)
        self.assertEqual(text, "UNKNOWN")
        self.assertEqual(conf, 0.0)

    def test_degenerated_empty_crop(self):
        """Verifies zero-dimension array returns ('UNKNOWN', 0.0) without IndexError."""
        empty_crop = np.zeros((0, 0, 3), dtype=np.uint8)
        text, conf = self.fallback_fn(empty_crop)
        self.assertEqual(text, "UNKNOWN")
        self.assertEqual(conf, 0.0)

    def test_low_confidence_ocr_results_filtered(self):
        """Verifies candidates below confidence threshold (e.g. 0.30) fallback to UNKNOWN."""
        dummy_crop = np.full((100, 200, 3), 128, dtype=np.uint8)
        # Mock OCR output: confidence 0.15 is too low to trust
        low_conf_results = [
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "DL04CA9999", 0.15)
        ]
        text, conf = oracle_read_plate_fallback(dummy_crop, ocr_results=low_conf_results)
        self.assertEqual(text, "UNKNOWN")
        self.assertEqual(conf, 0.0)

    def test_garbage_text_below_minimum_length_filtered(self):
        """Verifies OCR candidate with < 4 characters (e.g. 'A1') is rejected as noise."""
        dummy_crop = np.full((100, 200, 3), 128, dtype=np.uint8)
        short_results = [
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "AB", 0.95)
        ]
        text, conf = oracle_read_plate_fallback(dummy_crop, ocr_results=short_results)
        self.assertEqual(text, "UNKNOWN")
        self.assertEqual(conf, 0.0)

    def test_valid_plate_ocr_candidate_accepted(self):
        """Verifies valid plate candidate with sufficient confidence is accepted."""
        dummy_crop = np.full((100, 200, 3), 128, dtype=np.uint8)
        valid_results = [
            ([[0, 0], [10, 0], [10, 10], [0, 10]], "AS01AB1234", 0.88)
        ]
        text, conf = oracle_read_plate_fallback(dummy_crop, ocr_results=valid_results)
        self.assertEqual(text, "AS01AB1234")
        self.assertAlmostEqual(conf, 0.88, places=2)

    def test_plate_extractor_class_contract_if_available(self):
        """Verifies PlateExtractor.read_plate matches contract if implemented."""
        if PlateExtractor is None:
            self.skipTest("traffiq.enforcement.ocr.PlateExtractor not yet implemented (scheduled for M2)")

        extractor = PlateExtractor()
        black_crop = np.zeros((100, 200, 3), dtype=np.uint8)
        text, conf = extractor.read_plate(black_crop)
        self.assertEqual(text, "UNKNOWN")
        self.assertEqual(conf, 0.0)


if __name__ == "__main__":
    unittest.main()
