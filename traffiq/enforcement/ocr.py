"""
License plate OCR extraction, ROI isolation, and image enhancement for TRAFFIQ.
Extracts the lower horizontal region of the vehicle crop, applies CLAHE contrast
enhancement and bilateral denoising, runs EasyOCR, and normalizes candidate strings
with graceful fallback to ('UNKNOWN', 0.0).
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Sequence, Tuple, Union
import cv2
import numpy as np

# Standard Indian RTO registration regex (e.g., AS01AB1234, DL04CA9999)
INDIAN_RTO_REGEX = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")


def clean_plate_text(raw_text: Optional[str]) -> str:
    """
    Filter string to uppercase alphanumeric characters only.

    Args:
        raw_text: Uncleaned string from OCR output.

    Returns:
        Sanitized uppercase alphanumeric string.
    """
    if not raw_text:
        return ""
    return re.sub(r"[^A-Za-z0-9]", "", str(raw_text)).upper()


def is_valid_rto_plate(plate: str) -> bool:
    """
    Validate whether a plate string strictly complies with Indian RTO registration syntax.

    Args:
        plate: Normalized plate candidate string.

    Returns:
        True if the string matches the RTO pattern.
    """
    cleaned = clean_plate_text(plate)
    return bool(INDIAN_RTO_REGEX.match(cleaned))


def extract_plate_roi(
    vehicle_crop: np.ndarray,
    y_start_ratio: float = 0.55,
    y_end_ratio: float = 1.0,
) -> np.ndarray:
    """
    Isolate the lower 40-50% horizontal band of a vehicle bounding box crop
    where license plates are physically mounted, filtering out windshields,
    rooflines, and upper vehicle badges.

    Args:
        vehicle_crop: Cropped vehicle bounding box image array.
        y_start_ratio: Vertical start fraction of crop height (default 0.55).
        y_end_ratio: Vertical end fraction of crop height (default 1.0).

    Returns:
        The extracted sub-image slice with boundary clamping guaranteed.
    """
    if vehicle_crop is None or vehicle_crop.size == 0:
        return np.empty((0, 0, 3), dtype=np.uint8)

    h, w = vehicle_crop.shape[:2]
    if h == 0 or w == 0:
        return np.empty((0, 0, 3), dtype=np.uint8)

    # Calculate lower band bounds
    y1 = int(np.clip(h * y_start_ratio, 0, h))
    y2 = int(np.clip(h * y_end_ratio, 0, h))

    # Apply strict boundary clamping
    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))

    return vehicle_crop[y1:y2, :]


def clamp_bbox(
    bbox: Sequence[Union[int, float]],
    frame_shape: Tuple[int, ...],
) -> Tuple[int, int, int, int]:
    """
    Clamp bounding box coordinates [x1, y1, x2, y2] strictly within frame dimensions.

    Args:
        bbox: Sequence of [x1, y1, x2, y2].
        frame_shape: Target image shape tuple (height, width, ...).

    Returns:
        Clamped integer coordinates (cx1, cy1, cx2, cy2).
    """
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox[:4]
    cx1 = max(0, min(int(x1), w - 1))
    cy1 = max(0, min(int(y1), h - 1))
    cx2 = max(cx1 + 1, min(int(x2), w))
    cy2 = max(cy1 + 1, min(int(y2), h))
    return cx1, cy1, cx2, cy2


def preprocess_plate(roi: np.ndarray) -> np.ndarray:
    """
    Preprocess license plate ROI for OCR text extraction:
    1. Grayscale conversion (if multi-channel).
    2. CLAHE (Contrast Limited Adaptive Histogram Equalization) for localized contrast enhancement.
    3. Bilateral filter for edge-preserving denoising.

    Args:
        roi: License plate region of interest array.

    Returns:
        Preprocessed single-channel 2D uint8 numpy array.
    """
    if roi is None or roi.size == 0:
        return np.empty((0, 0), dtype=np.uint8)

    # 1. Grayscale conversion
    if len(roi.shape) == 3 and roi.shape[2] == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    elif len(roi.shape) == 2:
        gray = roi.copy()
    else:
        raise ValueError(f"Unsupported image shape for plate preprocessing: {roi.shape}")

    # 2. CLAHE Contrast Enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 3. Bilateral Filter for edge-preserving smoothing
    filtered = cv2.bilateralFilter(enhanced, d=5, sigmaColor=75, sigmaSpace=75)
    return filtered


def read_license_plate(
    vehicle_crop: np.ndarray,
    ocr_results: Optional[Sequence[Any]] = None,
    reader: Optional[object] = None,
) -> Tuple[str, float]:
    """
    Read license plate characters from vehicle crop with robust fallback to ('UNKNOWN', 0.0).

    Args:
        vehicle_crop: Vehicle image crop.
        ocr_results: Optional mocked/precomputed OCR result tuples (bbox, text, prob).
        reader: Optional EasyOCR reader instance.

    Returns:
        Tuple of (detected_plate_string, confidence_float).
    """
    if vehicle_crop is None or vehicle_crop.size == 0:
        return "UNKNOWN", 0.0

    # If explicit OCR results provided (e.g. from tests or prior inferences)
    if ocr_results is not None:
        candidates: List[Tuple[str, float]] = []
        for item in ocr_results:
            if len(item) >= 3:
                _, text, conf = item[:3]
                cleaned = clean_plate_text(str(text))
                c_conf = float(conf)
                if len(cleaned) >= 4 and c_conf >= 0.3:
                    candidates.append((cleaned, c_conf))

        if candidates:
            # Prefer standard RTO plate format matches
            rto_matches = [c for c in candidates if is_valid_rto_plate(c[0])]
            if rto_matches:
                best = max(rto_matches, key=lambda c: (len(c[0]), c[1]))
                return best[0], best[1]
            best = max(candidates, key=lambda c: (len(c[0]), c[1]))
            return best[0], best[1]

        return "UNKNOWN", 0.0

    # Fast blank / degenerate / noise check before executing heavy OCR
    if (
        np.all(vehicle_crop == 0)
        or np.all(vehicle_crop == 255)
        or vehicle_crop.shape[0] < 8
        or vehicle_crop.shape[1] < 15
        or float(np.std(vehicle_crop)) < 3.0
    ):
        return "UNKNOWN", 0.0

    # If reader is available, run OCR on preprocessed ROI
    if reader is not None:
        try:
            roi = extract_plate_roi(vehicle_crop)
            if roi.size == 0:
                return "UNKNOWN", 0.0
            preproc = preprocess_plate(roi)
            results = reader.readtext(preproc)
            return read_license_plate(vehicle_crop, ocr_results=results)
        except Exception:
            return "UNKNOWN", 0.0

    return "UNKNOWN", 0.0


class PlateExtractor:
    """
    License plate character extractor encapsulating ROI isolation,
    adaptive image preprocessing, and EasyOCR recognition with graceful fallbacks.
    """

    def __init__(
        self,
        reader: Optional[object] = None,
        gpu: bool = True,
        languages: Optional[Sequence[str]] = None,
        auto_init_reader: bool = False,
    ) -> None:
        """
        Initialize PlateExtractor.

        Args:
            reader: Optional pre-initialized EasyOCR Reader.
            gpu: Flag to request GPU acceleration for EasyOCR.
            languages: Language codes to load (defaults to ['en']).
            auto_init_reader: Whether to immediately instantiate EasyOCR in __init__.
        """
        self._reader = reader
        self.gpu = bool(gpu)
        self.languages = list(languages) if languages is not None else ["en"]

        if auto_init_reader and self._reader is None:
            self._init_reader()

    def _init_reader(self) -> None:
        """Instantiate EasyOCR Reader safely with CUDA availability verification."""
        try:
            import easyocr
            use_gpu = self.gpu
            if use_gpu:
                try:
                    import torch
                    if not torch.cuda.is_available():
                        use_gpu = False
                except ImportError:
                    use_gpu = False

            self._reader = easyocr.Reader(self.languages, gpu=use_gpu)
        except Exception:
            self._reader = None

    @property
    def reader(self) -> Optional[object]:
        """Lazy access property for EasyOCR reader."""
        if self._reader is None:
            self._init_reader()
        return self._reader

    def extract_roi(
        self,
        vehicle_crop: np.ndarray,
        y_start_ratio: float = 0.55,
        y_end_ratio: float = 1.0,
    ) -> np.ndarray:
        """Extract lower horizontal ROI band from vehicle crop."""
        return extract_plate_roi(
            vehicle_crop, y_start_ratio=y_start_ratio, y_end_ratio=y_end_ratio
        )

    def preprocess(self, roi: np.ndarray) -> np.ndarray:
        """Preprocess plate ROI using grayscale + CLAHE + bilateral filtering."""
        return preprocess_plate(roi)

    def read_plate(
        self,
        vehicle_crop: np.ndarray,
        ocr_results: Optional[Sequence[Any]] = None,
    ) -> Tuple[str, float]:
        """
        Extract and read license plate from vehicle crop.

        Args:
            vehicle_crop: Vehicle image crop.
            ocr_results: Optional precomputed OCR result tuples.

        Returns:
            (plate_number, confidence). Returns ('UNKNOWN', 0.0) on failure.
        """
        if ocr_results is not None:
            return read_license_plate(vehicle_crop, ocr_results=ocr_results)

        # Blank, black, white, flat, or degenerate crops fallback immediately
        if (
            vehicle_crop is None
            or vehicle_crop.size == 0
            or np.all(vehicle_crop == 0)
            or np.all(vehicle_crop == 255)
            or vehicle_crop.shape[0] < 8
            or vehicle_crop.shape[1] < 15
            or float(np.std(vehicle_crop)) < 3.0
        ):
            return "UNKNOWN", 0.0

        try:
            roi = self.extract_roi(vehicle_crop)
            if roi.size == 0:
                return "UNKNOWN", 0.0

            preprocessed = self.preprocess(roi)
            if self.reader is None:
                return "UNKNOWN", 0.0

            results = self.reader.readtext(preprocessed)
            if not results:
                # Fallback attempt on raw ROI if preprocessed returned no boxes
                results = self.reader.readtext(roi)

            return read_license_plate(vehicle_crop, ocr_results=results)
        except Exception:
            return "UNKNOWN", 0.0
