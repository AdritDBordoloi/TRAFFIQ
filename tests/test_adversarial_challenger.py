"""Adversarial stress-test suite for TRAFFIQ system boundary conditions.
Authored by teamwork_preview_challenger (Input & Boundary Challenger).

Covers:
1. Database & Serialization:
   - Passing np.int64, np.int32, np.uint64, np.int16 scalars into record_challan() -> stored as INTEGER, not BLOB.
   - decode_track_id_blob() with 8-byte, 4-byte, 2-byte, 1-byte, and corrupted byte buffers with filename fallback.
2. Plate Extraction & OCR:
   - extract_plate_roi() on extreme aspect ratios (tall, wide), small crops (10x10, 1x1), and boundary conditions.
   - read_plate() on pure black image, pure white image, random noise arrays, and degraded images -> returns ('UNKNOWN', 0.0).
3. Geometry & Math:
   - calculate_centroid() on zero-width/height and negative coordinates.
   - check_line_crossing() on reverse motion, horizontal movement, and boundary conditions.
4. Config:
   - TraffiqConfig with extreme valid thresholds (0.0, 1.0, 0 fine) and invalid thresholds (negative fine, out-of-bounds confidence).
"""

from __future__ import annotations

import os
import re
import shutil
import sqlite3
import struct
import tempfile
import unittest
from typing import Any

import numpy as np

# TRAFFIQ imports
from traffiq.config.settings import TraffiqConfig
from traffiq.database.connection import create_connection, register_sqlite_adapters
from traffiq.database.migration import decode_track_id_blob, migrate_legacy_blobs, verify_integrity
from traffiq.database.repository import DatabaseManager, record_challan
from traffiq.enforcement.ocr import PlateExtractor, clean_plate_text, extract_plate_roi, is_valid_rto_plate, read_license_plate
from traffiq.vision.tripwire import VirtualTripwire, calculate_centroid, check_line_crossing


class TestDatabaseAdversarial(unittest.TestCase):
    """Adversarial stress-testing of database serialization and blob recovery."""

    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "adversarial_test.db")
        register_sqlite_adapters()
        self.manager = DatabaseManager(self.db_path)

    def tearDown(self) -> None:
        self.manager.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_numpy_int64_scalar_in_record_challan(self) -> None:
        """Passing np.int64 into record_challan() must store as INTEGER, not BLOB."""
        track_id_np = np.int64(9876543210)
        challan_id = self.manager.record_challan(
            track_id=track_id_np,
            plate_number="AS01AB1234",
            owner_name="Test Owner",
            violation_type="Red Light Jump",
            fine_amount=1000,
            evidence_path="violations/test_1.jpg",
        )
        self.assertIsInstance(challan_id, int)

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT track_id, typeof(track_id) FROM challan_records WHERE challan_id = ?", (challan_id,))
        row = cur.fetchone()
        conn.close()

        self.assertIsNotNone(row)
        self.assertEqual(row[0], 9876543210)
        self.assertEqual(row[1], "integer", f"Expected SQLite typeof integer, got {row[1]}")

    def test_numpy_int32_scalar_in_record_challan(self) -> None:
        """Passing np.int32 into record_challan() must store as INTEGER, not BLOB."""
        track_id_np = np.int32(123456)
        challan_id = self.manager.record_challan(
            track_id=track_id_np,
            plate_number="DL04CA9999",
            owner_name="Test Owner 32",
            violation_type="Red Light Jump",
            fine_amount=1500,
            evidence_path="violations/test_2.jpg",
        )
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT track_id, typeof(track_id) FROM challan_records WHERE challan_id = ?", (challan_id,))
        row = cur.fetchone()
        conn.close()

        self.assertIsNotNone(row)
        self.assertEqual(row[0], 123456)
        self.assertEqual(row[1], "integer")

    def test_numpy_uint64_scalar_in_record_challan(self) -> None:
        """Passing np.uint64 into record_challan() must store as INTEGER, not BLOB."""
        track_id_np = np.uint64(555555)
        challan_id = self.manager.record_challan(
            track_id=track_id_np,
            plate_number="MH02DZ4567",
            owner_name="Test Owner U64",
            violation_type="Speeding",
            fine_amount=2000,
            evidence_path="violations/test_3.jpg",
        )
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT track_id, typeof(track_id) FROM challan_records WHERE challan_id = ?", (challan_id,))
        row = cur.fetchone()
        conn.close()

        self.assertIsNotNone(row)
        self.assertEqual(row[0], 555555)
        self.assertEqual(row[1], "integer")

    def test_numpy_int16_scalar_in_record_challan(self) -> None:
        """Passing np.int16 into record_challan() must store as INTEGER, not BLOB."""
        track_id_np = np.int16(32760)
        challan_id = self.manager.record_challan(
            track_id=track_id_np,
            plate_number="KA05MJ8821",
            owner_name="Test Owner 16",
            violation_type="Stop Line",
            fine_amount=500,
            evidence_path="violations/test_4.jpg",
        )
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT track_id, typeof(track_id) FROM challan_records WHERE challan_id = ?", (challan_id,))
        row = cur.fetchone()
        conn.close()

        self.assertIsNotNone(row)
        self.assertEqual(row[0], 32760)
        self.assertEqual(row[1], "integer")

    def test_module_level_record_challan_with_numpy_scalar(self) -> None:
        """Module-level record_challan() function handles numpy integers safely."""
        cid = record_challan(
            track_id=np.int64(777),
            plate_number="AS01AB1234",
            owner_name="Module Level",
            db_path=self.db_path,
        )
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT track_id, typeof(track_id) FROM challan_records WHERE challan_id = ?", (cid,))
        row = cur.fetchone()
        conn.close()
        self.assertEqual(row[0], 777)
        self.assertEqual(row[1], "integer")

    def test_decode_track_id_blob_8_byte(self) -> None:
        """8-byte little endian int64 buffer must decode properly."""
        buffer = struct.pack("<q", 123456789)
        decoded = decode_track_id_blob(buffer, evidence_path="")
        self.assertEqual(decoded, 123456789)

    def test_decode_track_id_blob_4_byte(self) -> None:
        """4-byte little endian int32 buffer must decode properly."""
        buffer = struct.pack("<i", 424242)
        decoded = decode_track_id_blob(buffer, evidence_path="")
        self.assertEqual(decoded, 424242)

    def test_decode_track_id_blob_2_byte_and_1_byte(self) -> None:
        """2-byte and 1-byte buffers decode properly."""
        buf2 = struct.pack("<h", 1024)
        self.assertEqual(decode_track_id_blob(buf2), 1024)

        buf1 = bytes([99])
        self.assertEqual(decode_track_id_blob(buf1), 99)

    def test_decode_track_id_blob_corrupted_with_filename_fallback(self) -> None:
        """Corrupted byte buffer with valid evidence_path filename falls back to regex."""
        # Non-standard buffer length or negative decoded value
        corrupt_buf = b"\x00\x00\x00\x00"  # unpacks to 0 which is non-positive
        evidence_path = "violations/violation_ID8821_20261004.jpg"
        decoded = decode_track_id_blob(corrupt_buf, evidence_path=evidence_path)
        self.assertEqual(decoded, 8821)

        # Arbitrary corrupted buffer of odd length
        odd_corrupt_buf = b"bad\xff\x00"
        decoded2 = decode_track_id_blob(odd_corrupt_buf, evidence_path="violations/violation_ID4321_snap.jpg")
        self.assertEqual(decoded2, 4321)

    def test_decode_track_id_blob_corrupted_without_fallback_raises_valueerror(self) -> None:
        """Corrupted buffer without recoverable evidence_path raises ValueError."""
        corrupt_buf = b"invalid"
        with self.assertRaises(ValueError):
            decode_track_id_blob(corrupt_buf, evidence_path="invalid_path.jpg")


class TestPlateExtractionAndOCRAdversarial(unittest.TestCase):
    """Adversarial stress-testing of Plate ROI and OCR fallback."""

    def test_extract_plate_roi_extreme_aspect_ratios(self) -> None:
        """Extreme aspect ratios (very wide or very tall) do not cause IndexError."""
        # Extremely wide crop: 10 height, 1000 width
        wide_crop = np.zeros((10, 1000, 3), dtype=np.uint8)
        roi_wide = extract_plate_roi(wide_crop)
        self.assertEqual(roi_wide.shape[1], 1000)
        self.assertGreater(roi_wide.shape[0], 0)
        self.assertLessEqual(roi_wide.shape[0], 10)

        # Extremely tall crop: 1000 height, 10 width
        tall_crop = np.zeros((1000, 10, 3), dtype=np.uint8)
        roi_tall = extract_plate_roi(tall_crop)
        self.assertEqual(roi_tall.shape[1], 10)
        self.assertGreater(roi_tall.shape[0], 0)
        self.assertLessEqual(roi_tall.shape[0], 1000)

    def test_extract_plate_roi_small_and_micro_crops(self) -> None:
        """10x10 and 1x1 crops must be clamped properly without empty slice or crash."""
        crop_10x10 = np.ones((10, 10, 3), dtype=np.uint8) * 128
        roi_10x10 = extract_plate_roi(crop_10x10)
        self.assertGreater(roi_10x10.shape[0], 0)
        self.assertEqual(roi_10x10.shape[1], 10)

        crop_1x1 = np.ones((1, 1, 3), dtype=np.uint8)
        roi_1x1 = extract_plate_roi(crop_1x1)
        self.assertEqual(roi_1x1.shape, (1, 1, 3))

    def test_extract_plate_roi_boundary_and_degenerate_conditions(self) -> None:
        """Empty, None, or zero-dimension inputs return safe empty array."""
        empty_crop = np.empty((0, 0, 3), dtype=np.uint8)
        roi_empty = extract_plate_roi(empty_crop)
        self.assertEqual(roi_empty.size, 0)

        roi_none = extract_plate_roi(None)
        self.assertEqual(roi_none.size, 0)

        # Inverted ratios: y_start > y_end
        crop = np.zeros((100, 200, 3), dtype=np.uint8)
        roi_inverted = extract_plate_roi(crop, y_start_ratio=0.9, y_end_ratio=0.1)
        self.assertGreater(roi_inverted.shape[0], 0)
        self.assertEqual(roi_inverted.shape[1], 200)

    def test_read_plate_pure_black_image(self) -> None:
        """Pure black frame must deterministically return ('UNKNOWN', 0.0)."""
        extractor = PlateExtractor(auto_init_reader=False)
        black_crop = np.zeros((120, 240, 3), dtype=np.uint8)
        text, conf = extractor.read_plate(black_crop)
        self.assertEqual(text, "UNKNOWN")
        self.assertEqual(conf, 0.0)

        # Direct function call
        text2, conf2 = read_license_plate(black_crop)
        self.assertEqual(text2, "UNKNOWN")
        self.assertEqual(conf2, 0.0)

    def test_read_plate_pure_white_image(self) -> None:
        """Pure white frame must deterministically return ('UNKNOWN', 0.0)."""
        extractor = PlateExtractor(auto_init_reader=False)
        white_crop = np.full((120, 240, 3), 255, dtype=np.uint8)
        text, conf = extractor.read_plate(white_crop)
        self.assertEqual(text, "UNKNOWN")
        self.assertEqual(conf, 0.0)

        text2, conf2 = read_license_plate(white_crop)
        self.assertEqual(text2, "UNKNOWN")
        self.assertEqual(conf2, 0.0)

    def test_read_plate_random_noise_arrays(self) -> None:
        """Random uniform noise images return ('UNKNOWN', 0.0)."""
        extractor = PlateExtractor(auto_init_reader=False)
        for seed in [42, 100, 999]:
            np.random.seed(seed)
            noise_crop = np.random.randint(0, 256, (120, 240, 3), dtype=np.uint8)
            text, conf = extractor.read_plate(noise_crop)
            self.assertEqual(text, "UNKNOWN")
            self.assertEqual(conf, 0.0)

    def test_read_plate_degraded_and_uniform_arrays(self) -> None:
        """Uniform gray, tiny, or low standard deviation images return ('UNKNOWN', 0.0)."""
        extractor = PlateExtractor(auto_init_reader=False)

        # Flat gray (std == 0.0)
        gray_crop = np.full((100, 200, 3), 127, dtype=np.uint8)
        text, conf = extractor.read_plate(gray_crop)
        self.assertEqual(text, "UNKNOWN")
        self.assertEqual(conf, 0.0)

        # Degenerate tiny crops
        tiny_crop = np.ones((5, 10, 3), dtype=np.uint8) * 100
        text2, conf2 = extractor.read_plate(tiny_crop)
        self.assertEqual(text2, "UNKNOWN")
        self.assertEqual(conf2, 0.0)


class TestGeometryAndMathAdversarial(unittest.TestCase):
    """Adversarial stress-testing of centroid calculation and line crossing."""

    def test_calculate_centroid_zero_width_and_height(self) -> None:
        """Zero-width and zero-height points must return exact coordinate."""
        bbox_point = (150, 250, 150, 250)
        cx, cy = calculate_centroid(bbox_point)
        self.assertEqual(cx, 150)
        self.assertEqual(cy, 250)

    def test_calculate_centroid_negative_coordinates(self) -> None:
        """Bounding boxes with negative coordinates must compute correct integer centroid."""
        bbox_neg = (-100, -200, -50, -100)
        cx, cy = calculate_centroid(bbox_neg)
        self.assertEqual(cx, -75)
        self.assertEqual(cy, -150)

        # Mixed negative and positive
        bbox_mixed = (-40, -60, 40, 60)
        cx2, cy2 = calculate_centroid(bbox_mixed)
        self.assertEqual(cx2, 0)
        self.assertEqual(cy2, 0)

    def test_calculate_centroid_floats_and_truncation(self) -> None:
        """Floating point box coordinates truncate correctly to integers."""
        bbox_float = (10.9, 20.1, 30.1, 40.9)
        cx, cy = calculate_centroid(bbox_float)
        self.assertEqual(cx, 20)
        self.assertEqual(cy, 30)

    def test_check_line_crossing_reverse_motion(self) -> None:
        """Reverse motion (moving up when direction='down', or down when direction='up') is rejected."""
        line_y = 400

        # Moving from 450 to 350 (upwards) when direction='down'
        self.assertFalse(check_line_crossing(prev_cy=450, curr_cy=350, line_y=line_y, direction="down"))

        # Moving from 350 to 450 (downwards) when direction='up'
        self.assertFalse(check_line_crossing(prev_cy=350, curr_cy=450, line_y=line_y, direction="up"))

    def test_check_line_crossing_horizontal_movement(self) -> None:
        """Horizontal movement with constant y must never trigger crossing."""
        line_y = 400

        # Moving horizontally above the line
        self.assertFalse(check_line_crossing(prev_cy=200, curr_cy=200, line_y=line_y, direction="down"))

        # Moving horizontally below the line
        self.assertFalse(check_line_crossing(prev_cy=600, curr_cy=600, line_y=line_y, direction="down"))

        # Moving horizontally exactly along the line
        self.assertFalse(check_line_crossing(prev_cy=line_y, curr_cy=line_y, line_y=line_y, direction="down"))

    def test_check_line_crossing_boundary_conditions(self) -> None:
        """Boundary conditions: starting on line, ending on line, sub-pixel/1-pixel jumps."""
        line_y = 400

        # Starts strictly above, lands exactly on line -> triggers downward crossing
        self.assertTrue(check_line_crossing(prev_cy=399, curr_cy=400, line_y=line_y, direction="down"))

        # Starts on line, moves past line -> does NOT trigger downward crossing (p_cy < l_y is False)
        self.assertFalse(check_line_crossing(prev_cy=400, curr_cy=401, line_y=line_y, direction="down"))

        # Upward crossing: starts strictly below, lands on line -> triggers upward crossing
        self.assertTrue(check_line_crossing(prev_cy=401, curr_cy=400, line_y=line_y, direction="up"))

        # Starts on line, moves above -> does NOT trigger upward crossing (p_cy > l_y is False)
        self.assertFalse(check_line_crossing(prev_cy=400, curr_cy=399, line_y=line_y, direction="up"))

        # Invalid direction string returns False safely without crash
        self.assertFalse(check_line_crossing(prev_cy=300, curr_cy=500, line_y=line_y, direction="diagonal"))


class TestConfigAdversarial(unittest.TestCase):
    """Adversarial stress-testing of TraffiqConfig boundary and error conditions."""

    def test_extreme_valid_thresholds(self) -> None:
        """Zero confidence, 1.0 confidence, zero fine, and fractional tripwire ratios are valid."""
        cfg_min = TraffiqConfig(
            confidence_threshold=0.0,
            tripwire_ratio=0.0,
            fine_amount=0,
            signal_duration=0.1,
        )
        self.assertEqual(cfg_min.confidence_threshold, 0.0)
        self.assertEqual(cfg_min.tripwire_ratio, 0.0)
        self.assertEqual(cfg_min.fine_amount, 0)
        self.assertEqual(cfg_min.signal_duration, 0.1)

        cfg_max = TraffiqConfig(
            confidence_threshold=1.0,
            tripwire_ratio=1.0,
            fine_amount=1_000_000,
            signal_duration=3600.0,
        )
        self.assertEqual(cfg_max.confidence_threshold, 1.0)
        self.assertEqual(cfg_max.tripwire_ratio, 1.0)
        self.assertEqual(cfg_max.fine_amount, 1_000_000)

    def test_invalid_negative_fine_amount_raises_valueerror(self) -> None:
        """Negative fine amounts must raise ValueError."""
        with self.assertRaises(ValueError):
            TraffiqConfig(fine_amount=-1)

        with self.assertRaises(ValueError):
            TraffiqConfig(fine_amount=-1000)

    def test_invalid_confidence_threshold_raises_valueerror(self) -> None:
        """Out-of-bounds confidence thresholds (< 0.0 or > 1.0) must raise ValueError."""
        with self.assertRaises(ValueError):
            TraffiqConfig(confidence_threshold=-0.05)

        with self.assertRaises(ValueError):
            TraffiqConfig(confidence_threshold=1.01)

        with self.assertRaises(ValueError):
            TraffiqConfig(confidence_threshold=2.5)

    def test_invalid_tripwire_ratio_raises_valueerror(self) -> None:
        """Out-of-bounds tripwire ratio (< 0.0 or > 1.0) must raise ValueError."""
        with self.assertRaises(ValueError):
            TraffiqConfig(tripwire_ratio=-0.1)

        with self.assertRaises(ValueError):
            TraffiqConfig(tripwire_ratio=1.1)

    def test_invalid_signal_duration_raises_valueerror(self) -> None:
        """Zero or negative signal durations must raise ValueError."""
        with self.assertRaises(ValueError):
            TraffiqConfig(signal_duration=0.0)

        with self.assertRaises(ValueError):
            TraffiqConfig(signal_duration=-5.0)

    def test_invalid_max_frames_raises_valueerror(self) -> None:
        """Zero or negative max_frames must raise ValueError."""
        with self.assertRaises(ValueError):
            TraffiqConfig(max_frames=0)

        with self.assertRaises(ValueError):
            TraffiqConfig(max_frames=-10)

    def test_camera_index_coercion_adversarial(self) -> None:
        """String camera index with whitespace and digits coerces to int, string path remains str."""
        cfg_str_digit = TraffiqConfig(camera_index=" 2 ")
        self.assertEqual(cfg_str_digit.camera_index, 2)

        cfg_str_path = TraffiqConfig(camera_index="rtsp://192.168.1.1:554/live")
        self.assertEqual(cfg_str_path.camera_index, "rtsp://192.168.1.1:554/live")


if __name__ == "__main__":
    unittest.main()
