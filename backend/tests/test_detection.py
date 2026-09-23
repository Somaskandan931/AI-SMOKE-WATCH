def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["yolo_mode"] in ("model", "mock")


def test_detect_valid_image(client, plain_gray_jpeg_bytes):
    r = client.post(
        "/api/detect",
        files={"image": ("car.jpg", plain_gray_jpeg_bytes, "image/jpeg")},
    )
    assert r.status_code == 200
    body = r.json()
    assert "vehicle_detected" in body
    assert "smoke_detected" in body
    assert "reporting_allowed" in body
    assert isinstance(body["reporting_allowed"], bool)
    assert body["mode"] in ("model", "mock")


def test_detect_invalid_image_type(client):
    r = client.post(
        "/api/detect",
        files={"image": ("notanimage.txt", b"hello world", "text/plain")},
    )
    assert r.status_code == 422


def test_detect_empty_file(client):
    r = client.post(
        "/api/detect",
        files={"image": ("empty.jpg", b"", "image/jpeg")},
    )
    assert r.status_code == 422


def test_reporting_gate_flat_gray_image_has_no_smoke(client, plain_gray_jpeg_bytes):
    """A flat, uniform-color image should not have vehicle/smoke edges to
    detect, so the reporting gate must stay disabled (FR-05 negative case)."""
    r = client.post(
        "/api/detect",
        files={"image": ("flat.jpg", plain_gray_jpeg_bytes, "image/jpeg")},
    )
    body = r.json()
    if body["mode"] == "mock":
        assert body["reporting_allowed"] is False
