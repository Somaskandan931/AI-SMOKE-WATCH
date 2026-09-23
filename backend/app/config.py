"""
Central configuration for the backend.

MOCK_INFERENCE controls whether the app uses a real trained YOLO model
(ai/weights/best.pt) or a deterministic, clearly-labelled mock detector.
The app auto-detects: if a weights file exists AND ultralytics is
installed, it uses real inference. Otherwise it falls back to mock mode
so the rest of the pipeline (backend, Flutter app, report flow) can
still be demoed end-to-end without a trained model or GPU.
"""

import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
AI_DIR = PROJECT_ROOT / "ai"
WEIGHTS_DIR = AI_DIR / "weights"

# Two real, separately-sourced models stand in for the single unified
# vehicle+exhaust_smoke model the original spec describes. No dataset of
# vehicles-with-exhaust-smoke exists to train that unified model, so:
#   - VEHICLE: a genuine pretrained Ultralytics YOLOv8n (COCO), which
#     already has car/truck/bus/motorcycle classes — no training needed.
#   - SMOKE: a genuine fine-tuned YOLOv8s trained on general fire/smoke
#     imagery (not exhaust-specific). This is real inference with a real,
#     if domain-transferred, model — not a placeholder.
# Swapping in a proper exhaust-smoke-trained model later only means
# replacing SMOKE_WEIGHTS_FILENAME's file; no other code changes.
VEHICLE_WEIGHTS_FILENAME = "vehicle_yolov8n.pt"
SMOKE_WEIGHTS_FILENAME = "smoke_yolov8s.pt"
PLATE_WEIGHTS_FILENAME = "plate_best.pt"

# COCO class names (from the vehicle model) that count as "vehicle".
VEHICLE_CLASS_NAMES = {"car", "truck", "bus", "motorcycle"}
# Class name (from the smoke model) that counts as exhaust smoke. The
# model's other classes, "Fire" and a stray "default" (an annotation
# artifact in the source dataset with near-zero real detections), are
# deliberately ignored — this app detects smoke, not fire. Note the
# lowercase "smoke": this is the exact class name as trained into the
# downloaded weights (see ai/weights/README.md for provenance) — verify
# against model.names if you ever swap the weights file.
SMOKE_CLASS_NAME = "smoke"


def find_vehicle_weights() -> Path | None:
    candidate = WEIGHTS_DIR / VEHICLE_WEIGHTS_FILENAME
    return candidate if candidate.exists() else None


def find_smoke_weights() -> Path | None:
    candidate = WEIGHTS_DIR / SMOKE_WEIGHTS_FILENAME
    return candidate if candidate.exists() else None


def find_plate_weights() -> Path | None:
    candidate = WEIGHTS_DIR / PLATE_WEIGHTS_FILENAME
    return candidate if candidate.exists() else None


def find_trained_weights() -> Path | None:
    """Back-compat single-model lookup (best.pt/last.pt), kept in case a
    future unified vehicle+exhaust_smoke model is dropped in — see
    yolo_service.py for how it takes priority over the two-model path."""
    for name in ["best.pt", "last.pt"]:
        candidate = WEIGHTS_DIR / name
        if candidate.exists():
            return candidate
    return None


def ultralytics_available() -> bool:
    try:
        import ultralytics  # noqa: F401

        return True
    except ImportError:
        return False


def easyocr_available() -> bool:
    try:
        import easyocr  # noqa: F401

        return True
    except ImportError:
        return False


def pytesseract_available() -> bool:
    """Tesseract is the OCR engine actually verified working in this
    project's dev sandbox (apt-installable, no large model download from
    a host outside the environment's allowed network). EasyOCR is
    supported too (see ocr_service.py) and is preferred if its models are
    already cached, but pytesseract is the more reliably available real
    engine and is tried second, before falling back to mock."""
    try:
        import pytesseract  # noqa: F401
        import shutil

        return shutil.which("tesseract") is not None
    except ImportError:
        return False


# Tests force mock mode (see backend/tests/conftest.py) so the suite stays
# fast, deterministic, and independent of ~600MB of model weights — this
# is the standard reason unit tests stub out ML inference. Production and
# manual/demo runs use real inference by default whenever the weights and
# libraries are present.
FORCE_MOCK_INFERENCE = os.getenv("FORCE_MOCK_INFERENCE", "0") == "1"


# Detection thresholds — tune these experimentally against a validation set.
SMOKE_CONFIDENCE_THRESHOLD = float(os.getenv("SMOKE_CONFIDENCE_THRESHOLD", "0.5"))
VEHICLE_CONFIDENCE_THRESHOLD = float(os.getenv("VEHICLE_CONFIDENCE_THRESHOLD", "0.5"))

# How much a smoke box is allowed to sit outside the vehicle box (as a
# fraction of the smoke box's own area) and still count as "associated".
# Smoke plumes drift up/behind the vehicle, so we pad the vehicle box
# rather than requiring strict containment.
ASSOCIATION_PADDING_RATIO = 0.35
MIN_ASSOCIATION_OVERLAP = 0.15

_unified_model_available = find_trained_weights() is not None and ultralytics_available()
_two_model_available = (
    find_vehicle_weights() is not None
    and find_smoke_weights() is not None
    and ultralytics_available()
)
MOCK_MODE = FORCE_MOCK_INFERENCE or not (_unified_model_available or _two_model_available)

# Where uploaded images are temporarily written for processing.
UPLOAD_DIR = BACKEND_DIR / "tmp_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

CIVIC_AUTHORITY_HANDLE = os.getenv("CIVIC_AUTHORITY_HANDLE", "@ChennaiTrafficP")

# X (Twitter) API credentials for actually publishing reports (see
# app/services/x_client.py). All four are OAuth 1.0a user-context
# credentials from a single X developer app -- generate once in the
# developer portal under Keys and tokens, no login flow required.
# Leave any of these unset to keep report/publish in mock mode (see
# report_service.publish_report_to_x), same auto-detect pattern used for
# ML inference and OCR above.
X_API_KEY = os.getenv("X_API_KEY")
X_API_SECRET = os.getenv("X_API_SECRET")
X_ACCESS_TOKEN = os.getenv("X_ACCESS_TOKEN")
X_ACCESS_SECRET = os.getenv("X_ACCESS_SECRET")


def x_credentials_configured() -> bool:
    return all([X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, X_ACCESS_SECRET])
