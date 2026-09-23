"""
License plate OCR (PRD section 14.8 / FR-09, FR-10).

Tries EasyOCR first (pure-Python, easier to install than PaddleOCR), then
PaddleOCR, and if neither is installed/available, returns
`ocr_available=False` with `registration_number=None` -- the API contract the
Flutter app already expects for "OCR failed, let the user type it in"
(FR-10 / non-functional Reliability requirement).

This module is intentionally decoupled from *which* OCR engine is installed:
swap in PaddleOCR by installing `paddleocr` and `paddlepaddle`; no other code
needs to change.
"""
import re
from typing import Optional, Tuple

import numpy as np

from app import config
from app.utils.validation import looks_like_valid_plate

_engine = None
_engine_name = None


def _load_engine():
    global _engine, _engine_name
    if _engine is not None or _engine_name is not None:
        return
    try:
        import easyocr

        _engine = easyocr.Reader(["en"], gpu=False)
        _engine_name = "easyocr"
        return
    except Exception:
        pass
    try:
        from paddleocr import PaddleOCR

        _engine = PaddleOCR(use_angle_cls=True, lang="en")
        _engine_name = "paddleocr"
        return
    except Exception:
        pass
    _engine_name = "unavailable"


PLATE_PATTERN = re.compile(r"[^A-Z0-9]")


def _clean(text: str) -> str:
    cleaned = PLATE_PATTERN.sub("", text.upper())
    # HSRP plates carry an "IND" marking that OCR reads as part of the number.
    if cleaned.startswith("IND") and looks_like_valid_plate(cleaned[3:]):
        cleaned = cleaned[3:]
    return cleaned


def read_plate(cropped_bgr: np.ndarray) -> Tuple[Optional[str], float, bool]:
    """Returns (registration_number, confidence, ocr_available)."""
    # Same FORCE_MOCK_INFERENCE contract as yolo_service/plate_detector:
    # deterministic, engine-independent output so the test suite (and a
    # demo with no OCR engine installed) doesn't depend on a real OCR
    # engine being present. Without this, OCR was the one stage in the
    # pipeline that couldn't be demoed in mock mode -- it just always fell
    # through to "unavailable" -- which broke the documented end-to-end
    # demo flow whenever easyocr/paddleocr weren't installed.
    if config.FORCE_MOCK_INFERENCE:
        return "TN38AB1234", 0.93, True

    _load_engine()

    if _engine_name == "unavailable" or _engine is None:
        return None, 0.0, False

    try:
        if _engine_name == "easyocr":
            results = _engine.readtext(cropped_bgr)
            if not results:
                return None, 0.0, True
            text = "".join(r[1] for r in results)
            confidences = [r[2] for r in results]
            return _clean(text), float(sum(confidences) / len(confidences)), True

        if _engine_name == "paddleocr":
            results = _engine.ocr(cropped_bgr, cls=True)
            lines = results[0] if results else []
            if not lines:
                return None, 0.0, True
            text = "".join(line[1][0] for line in lines)
            confidences = [line[1][1] for line in lines]
            return _clean(text), float(sum(confidences) / len(confidences)), True
    except Exception:
        return None, 0.0, True

    return None, 0.0, True
