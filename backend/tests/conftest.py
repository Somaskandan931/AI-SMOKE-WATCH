import io
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Force mock inference for the whole test run -- keeps tests fast,
# deterministic, and independent of ~30MB of model weights actually being
# present/loadable. Must be set before app.config (and anything that
# imports it) is first imported, hence this sits at the top of conftest.py.
os.environ["FORCE_MOCK_INFERENCE"] = "1"


def make_test_jpeg(size=(300, 300), color=(120, 120, 120)) -> bytes:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from app.main import app

    return TestClient(app)


@pytest.fixture
def plain_gray_jpeg_bytes():
    return make_test_jpeg()


def _encode(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def smoky_image_bytes():
    """
    Built to satisfy the mock detector's heuristic in
    app/services/yolo_service.py::_detect_mock:
      - a dark rectangle in the lower two-thirds -> "vehicle" contour
      - a light, low-saturation grey band directly above it -> "smoke" haze
    White elsewhere stays out of the haze mask (val=255 fails val<235).
    """
    w, h = 300, 300
    img = Image.new("RGB", (w, h), color=(255, 255, 255))
    pixels = img.load()
    # Vehicle: dark rectangle, rows 180-280, cols 50-250.
    for y in range(180, 280):
        for x in range(50, 250):
            pixels[x, y] = (25, 25, 25)
    # Smoke haze: light grey band, rows 75-180, full width.
    for y in range(75, 180):
        for x in range(0, w):
            pixels[x, y] = (200, 200, 200)
    return _encode(img)


@pytest.fixture
def clean_image_bytes():
    """Same vehicle contour as smoky_image_bytes but no haze band above it,
    so the mock detector reports vehicle_detected=True, smoke_detected=False."""
    w, h = 300, 300
    img = Image.new("RGB", (w, h), color=(255, 255, 255))
    pixels = img.load()
    for y in range(180, 280):
        for x in range(50, 250):
            pixels[x, y] = (25, 25, 25)
    return _encode(img)


@pytest.fixture
def tiny_plate_bytes():
    """Content doesn't matter under FORCE_MOCK_INFERENCE=1 -- ocr_service
    and plate_detector both return deterministic mock results regardless
    of image content -- but it still has to be a valid, decodable JPEG."""
    return make_test_jpeg(size=(120, 40), color=(210, 210, 210))
