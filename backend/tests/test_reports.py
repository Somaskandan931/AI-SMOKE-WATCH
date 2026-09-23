def test_generate_report_success(client):
    r = client.post(
        "/api/report/generate",
        json={
            "registration_number": "tn38ab1234",
            "smoke_detected": True,
            "smoke_confidence": 0.91,
            "latitude": 13.0827,
            "longitude": 80.2707,
            "place_name": "Anna Salai, Chennai",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["registration"] == "TN38AB1234"
    assert "Suspected visible exhaust smoke" in body["report_text"]
    # Must never claim a confirmed legal violation (PRD 14.11).
    assert "violat" not in body["report_text"].lower()
    assert body["authority_handle"] in body["report_text"]
    assert body["x_intent_url"].startswith("https://twitter.com/intent/tweet?")


def test_generate_report_without_smoke_is_rejected(client):
    r = client.post(
        "/api/report/generate",
        json={
            "registration_number": "TN38AB1234",
            "smoke_detected": False,
            "smoke_confidence": 0.1,
        },
    )
    assert r.status_code == 400


def test_generate_report_missing_location_permission_falls_back(client):
    r = client.post(
        "/api/report/generate",
        json={
            "registration_number": "TN38AB1234",
            "smoke_detected": True,
            "smoke_confidence": 0.8,
        },
    )
    assert r.status_code == 200
    assert r.json()["location"] == "Location unavailable"


def test_generate_report_blank_registration_rejected(client):
    r = client.post(
        "/api/report/generate",
        json={
            "registration_number": "   ",
            "smoke_detected": True,
            "smoke_confidence": 0.8,
        },
    )
    assert r.status_code == 422
