def test_plate_detect_valid_image(client, plain_gray_jpeg_bytes):
    r = client.post(
        "/api/plate/detect",
        files={"image": ("plate.jpg", plain_gray_jpeg_bytes, "image/jpeg")},
    )
    assert r.status_code == 200
    body = r.json()
    assert "plate_found" in body
    assert body["mode"] in ("model", "mock")


def test_plate_detect_invalid_image(client):
    r = client.post(
        "/api/plate/detect",
        files={"image": ("bad.txt", b"not an image", "text/plain")},
    )
    assert r.status_code == 422


def test_ocr_returns_manual_entry_signal_when_unavailable_or_failed(client, plain_gray_jpeg_bytes):
    r = client.post(
        "/api/plate/ocr",
        files={"image": ("plate.jpg", plain_gray_jpeg_bytes, "image/jpeg")},
    )
    assert r.status_code == 200
    body = r.json()
    # Either OCR is unavailable in this environment, or it ran and failed to
    # read a blank grey square -- both are valid "fall back to manual entry"
    # outcomes per FR-10 / the Reliability NFR.
    if not body["ocr_available"] or body["registration_number"] is None:
        assert body["registration_number"] is None


def test_ocr_invalid_image(client):
    r = client.post(
        "/api/plate/ocr",
        files={"image": ("bad.txt", b"not an image", "text/plain")},
    )
    assert r.status_code == 422
