# Architecture

```
Flutter Mobile App (mobile/)
        |
        |  multipart image upload / JSON
        v
FastAPI Backend (backend/)
        |
        v
YoloService  --------------------->  ai/weights/best.pt (if present)
  |  no weights found?                     |
  |  -> OpenCV heuristic (mock mode)        v
  |                                   Ultralytics YOLO inference
  v
vehicle_detected, smoke_detected, spatial association, confidence gate
        |
        v
PlateDetector -> crop -> OcrService (EasyOCR/PaddleOCR, or "unavailable")
        |
        v
ReportService -> structured "suspected issue" text + X share-intent URL
        |
        v
Flutter Report Preview -> user taps "Report on X" -> X app/website
                           (user reviews & posts themselves — nothing is
                            auto-published)
```

## Why mock mode exists

The PRD explicitly separates "must work" from "can be simplified" for a
hackathon MVP, and says: *"If there is no trained model, create the
training/inference structure and a clearly documented placeholder/mock
inference mode."*

`YoloService` and `PlateDetector` both check for a weights file on startup:

- Found + loads correctly → `mode: "model"`, real Ultralytics YOLO inference.
- Missing or fails to load → `mode: "mock"`, an OpenCV heuristic (contour
  detection for the vehicle silhouette, low-saturation/high-brightness haze
  detection for smoke, spatially gated to sit above/behind the vehicle).

Every API response includes `mode` so the mobile app can show "demo mode"
in place of numbers that could be mistaken for a validated model's output —
this is called out on the AI Analysis screen (`SmokeStatusCard`).

## Reporting gate (FR-05, FR-06, section 14.9)

```
IF vehicle_detected
   AND smoke_detected
   AND smoke_confidence >= SMOKE_CONFIDENCE_THRESHOLD
   AND smoke_box spatially associated with vehicle_box
THEN reporting_allowed = true
ELSE reporting_allowed = false
```

The spatial-association check (`YoloService.boxes_associated`) expands the
vehicle's box by a margin (wider sideways, taller upward, to allow for a
plume trailing behind/above a tailpipe) and requires the smoke box to
overlap that expanded region — so a hazy patch of sky elsewhere in frame
does not enable reporting for an unrelated vehicle.

## Data flow through the mobile app

`VehicleReport` (mobile/lib/models/vehicle_report.dart) is a single mutable
object threaded through every screen via constructor injection, accumulating
the captured image, detection result, plate image/OCR result, location, and
finally the generated report text — mirroring the PRD's step-by-step flow
(section 7) one field at a time.

## Location & landmark resolution

`LocationService.getCurrentPosition()` uses `geolocator` (device GPS/network
location, no API key). `LocationService.reverseGeocode()` calls OpenStreetMap
Nominatim (free, no key) to turn coordinates into a landmark/road/locality
name rather than showing raw latitude/longitude, with a graceful fallback to
coordinates if the network call fails. See the comments in
`mobile/lib/core/services/location_service.dart` for how to switch to the
Google Maps Geocoding API if you obtain a key later.
