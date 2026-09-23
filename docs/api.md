# API Reference

Base URL (local dev): `http://localhost:8000/api`

## `GET /health`

```json
{ "status": "ok", "yolo_mode": "mock" }
```

## `POST /detect`

Multipart form field: `image` (jpeg/png/webp, ≤15MB).

```json
{
  "vehicle_detected": true,
  "smoke_detected": true,
  "vehicle_confidence": 0.96,
  "smoke_confidence": 0.91,
  "smoke_associated_with_vehicle": true,
  "reporting_allowed": true,
  "detections": [
    {"label": "vehicle", "confidence": 0.96, "box": {"x1": 12, "y1": 40, "x2": 300, "y2": 220}},
    {"label": "exhaust_smoke", "confidence": 0.91, "box": {"x1": 150, "y1": 10, "x2": 260, "y2": 60}}
  ],
  "mode": "model",
  "message": "Visible exhaust smoke detected."
}
```

Errors: `422` invalid/corrupt/oversized image. `500` if model inference
raises unexpectedly.

## `POST /plate/detect`

Multipart form field: `image`. Now runs OCR on the crop in the same call —
one upload instead of two.

```json
{
  "plate_found": true,
  "box": {"x1": 80, "y1": 210, "x2": 240, "y2": 250},
  "cropped_image_base64": "…",
  "registration_number": "TN38AB1234",
  "ocr_confidence": 0.94,
  "ocr_available": true,
  "mode": "mock",
  "message": "License plate located. Registration number extracted. Please verify it's correct."
}
```

If no OCR engine is installed, or OCR can't read the plate clearly:
`ocr_available: false` / `registration_number: null` — the client should
fall back to manual entry either way.

## `POST /plate/ocr` (optional — standalone re-OCR)

Multipart form field: `image` (full frame or a pre-cropped plate — the
endpoint runs plate detection internally if needed). Kept only for
re-running OCR on a plate crop you already have (e.g. the user manually
retook just the plate region); the normal flow no longer needs this
separate call since `/plate/detect` above already includes it. Safe to
delete — route, schema, and its two tests in `test_ocr.py` — if your
client only ever calls `/plate/detect`.

```json
{
  "registration_number": "TN38AB1234",
  "confidence": 0.94,
  "ocr_available": true,
  "mode": "mock",
  "message": "Registration number extracted. Please verify it's correct."
}
```

## `POST /report/generate`

```json
{
  "registration_number": "TN38AB1234",
  "smoke_detected": true,
  "smoke_confidence": 0.91,
  "latitude": 13.0827,
  "longitude": 80.2707,
  "place_name": "Anna Salai, Chennai",
  "timestamp": "2026-09-23T10:15:00Z"
}
```

```json
{
  "registration": "TN38AB1234",
  "location": "Anna Salai, Chennai",
  "timestamp": "23/09/2026, 03:45 PM",
  "smoke_detected": true,
  "smoke_confidence": 0.91,
  "report_text": "Suspected visible exhaust smoke was observed from a vehicle. …",
  "authority_handle": "@ChennaiTrafficP",
  "x_intent_url": "https://twitter.com/intent/tweet?text=..."
}
```

Errors: `400` if `smoke_detected` is false (the gate should already have
stopped the user before this screen). `422` for a blank registration number.
