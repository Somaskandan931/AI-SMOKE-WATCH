"""
End-to-end tests that exercise the full documented demo scenario
(README "Demo script") against the API layer, simulating what the
Flutter app does call-by-call.
"""


def test_e2e_positive_flow_smoke_detected_to_report(client, smoky_image_bytes, tiny_plate_bytes):
    # 1. Health check
    assert client.get("/api/health").status_code == 200

    # 2. Capture vehicle image -> detect
    detect_resp = client.post(
        "/api/detect",
        files={"image": ("vehicle.jpg", smoky_image_bytes, "image/jpeg")},
    )
    detect_body = detect_resp.json()
    assert detect_body["reporting_allowed"] is True

    # 3. Capture plate -> OCR
    ocr_resp = client.post(
        "/api/plate/ocr",
        files={"image": ("plate.jpg", tiny_plate_bytes, "image/jpeg")},
    )
    ocr_body = ocr_resp.json()
    assert ocr_body["registration_number"] != ""

    # 4. User verifies/edits registration, location + timestamp captured,
    #    then generate the report
    report_resp = client.post(
        "/api/report/generate",
        json={
            "registration_number": ocr_body["registration_number"],
            "smoke_detected": detect_body["smoke_detected"],
            "smoke_confidence": detect_body["smoke_confidence"],
            "latitude": 13.0827,
            "longitude": 80.2707,
            "location_name": "Chennai",
        },
    )
    assert report_resp.status_code == 200
    report_body = report_resp.json()

    # 5. Report is ready for user review + explicit "Report on X" action
    assert "twitter.com/intent/tweet" in report_body["x_intent_url"]
    assert ocr_body["registration_number"] in report_body["report_text"]
    assert "suspected" in report_body["report_text"].lower()


def test_e2e_negative_flow_no_smoke_blocks_reporting(client, clean_image_bytes):
    detect_resp = client.post(
        "/api/detect",
        files={"image": ("vehicle.jpg", clean_image_bytes, "image/jpeg")},
    )
    detect_body = detect_resp.json()
    assert detect_body["smoke_detected"] is False
    assert detect_body["reporting_allowed"] is False

    # Client must not be able to generate a report from this state.
    report_resp = client.post(
        "/api/report/generate",
        json={
            "registration_number": "TN38AB1234",
            "smoke_detected": detect_body["smoke_detected"],
            "smoke_confidence": detect_body["smoke_confidence"],
        },
    )
    assert report_resp.status_code == 400
