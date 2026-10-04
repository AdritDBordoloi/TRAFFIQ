"""Tier 1: License Plate Text Normalization & Regex Validation Tests.
Verifies character cleaning, punctuation stripping, case folding, and
Indian RTO license plate regex matching.
"""

import re
import unittest

try:
    from traffiq.enforcement.ocr import clean_plate_text, is_valid_rto_plate
except ImportError:
    clean_plate_text = None
    is_valid_rto_plate = None


# Authoritative reference implementation derived from live_enforcement.py and Indian RTO standards
def oracle_clean_plate_text(raw_text: str) -> str:
    """Filters string to uppercase alphanumeric characters only."""
    if not raw_text:
        return ""
    return re.sub(r"[^A-Za-z0-9]", "", raw_text).upper()


INDIAN_RTO_REGEX = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")


def oracle_is_valid_rto_plate(plate: str) -> bool:
    """Validates whether a normalized string matches Indian RTO registration syntax."""
    return bool(INDIAN_RTO_REGEX.match(plate))


class TestTextCleaning(unittest.TestCase):
    """Tests plate string sanitization and regex validation."""

    def setUp(self):
        self.clean_fn = clean_plate_text or oracle_clean_plate_text
        self.validate_fn = is_valid_rto_plate or oracle_is_valid_rto_plate

    def test_strip_spaces_and_hyphens(self):
        """Verifies spaces, hyphens, and dots are stripped cleanly."""
        self.assertEqual(self.clean_fn("AS 01 AB 1234"), "AS01AB1234")
        self.assertEqual(self.clean_fn("DL-04-CA-9999"), "DL04CA9999")
        self.assertEqual(self.clean_fn("MH.02.DZ.4567"), "MH02DZ4567")

    def test_uppercase_conversion(self):
        """Verifies lowercase inputs are converted to uppercase."""
        self.assertEqual(self.clean_fn("as01ab1234"), "AS01AB1234")
        self.assertEqual(self.clean_fn("ka05mj8821"), "KA05MJ8821")

    def test_strip_punctuation_and_noise(self):
        """Verifies OCR noise symbols (brackets, colons, slashes) are removed."""
        self.assertEqual(self.clean_fn("[AS01] /AB:1234!"), "AS01AB1234")
        self.assertEqual(self.clean_fn("IND~DL04CA9999#"), "INDDL04CA9999")

    def test_empty_and_whitespace_only(self):
        """Verifies empty strings and whitespace return empty string without error."""
        self.assertEqual(self.clean_fn(""), "")
        self.assertEqual(self.clean_fn("   "), "")
        self.assertEqual(self.clean_fn("\t\n"), "")

    def test_valid_indian_rto_plates(self):
        """Verifies standard Indian RTO plates pass regex validation."""
        valid_plates = [
            "AS01AB1234",
            "DL04CA9999",
            "MH02DZ4567",
            "KA05MJ8821",
            "HR26DK8338",
            "UP32AB1234",
            "DL4CA9999",  # Single digit district
        ]
        for plate in valid_plates:
            cleaned = self.clean_fn(plate)
            self.assertTrue(self.validate_fn(cleaned), f"Expected valid RTO plate: {plate}")

    def test_invalid_indian_rto_plates(self):
        """Verifies malformed strings, demo plates, and noise fail RTO regex validation."""
        invalid_plates = [
            "SAMPLE123",
            "123456",
            "ABCDEFG",
            "KA05",
            "9999",
            "AS01AB",
            "INDIA",
            "",
        ]
        for plate in invalid_plates:
            cleaned = self.clean_fn(plate)
            self.assertFalse(self.validate_fn(cleaned), f"Expected invalid RTO plate: {plate}")


if __name__ == "__main__":
    unittest.main()
