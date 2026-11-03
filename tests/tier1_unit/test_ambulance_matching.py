"""Tier 1: shared ambulance box-matching rule (traffiq.vision.ambulance).

Regression: the dashboard engine used to promote ANY tracked vehicle that overlapped an
ambulance detection by a single pixel, which falsely triggered the green corridor.
"""

import unittest

from traffiq.vision.ambulance import (
    compute_box_iomin,
    compute_box_iou,
    matches_ambulance_box,
    point_in_box,
)


class TestAmbulanceMatching(unittest.TestCase):
    AMB = (100, 100, 300, 250)

    def test_identical_boxes_match(self):
        self.assertTrue(matches_ambulance_box(self.AMB, self.AMB, 200, 175))
        self.assertAlmostEqual(compute_box_iou(self.AMB, self.AMB), 1.0)

    def test_one_pixel_overlap_does_not_match(self):
        neighbour = (299, 100, 500, 250)  # touches the ambulance box by 1px only
        self.assertGreater(compute_box_iou(neighbour, self.AMB), 0.0)
        self.assertFalse(matches_ambulance_box(neighbour, self.AMB, 400, 175))

    def test_disjoint_boxes_do_not_match(self):
        far = (600, 100, 800, 250)
        self.assertEqual(compute_box_iou(far, self.AMB), 0.0)
        self.assertEqual(compute_box_iomin(far, self.AMB), 0.0)
        self.assertFalse(matches_ambulance_box(far, self.AMB, 700, 175))

    def test_nested_small_box_matches_via_iomin(self):
        inner = (150, 120, 250, 220)  # fully inside the ambulance box
        self.assertEqual(compute_box_iomin(inner, self.AMB), 1.0)
        self.assertTrue(matches_ambulance_box(inner, self.AMB, 200, 170))

    def test_centroid_inside_matches(self):
        self.assertTrue(point_in_box(200, 175, self.AMB))
        self.assertFalse(point_in_box(50, 50, self.AMB))

    def test_degenerate_zero_area_box_is_safe(self):
        zero = (10, 10, 10, 10)
        self.assertEqual(compute_box_iou(zero, self.AMB), 0.0)
        self.assertEqual(compute_box_iomin(zero, self.AMB), 0.0)


if __name__ == "__main__":
    unittest.main()
