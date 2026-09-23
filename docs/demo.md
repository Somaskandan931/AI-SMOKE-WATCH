# Demo Script

## Setup

```bash
cd backend
pip install -r requirements.txt --break-system-packages   # or use a venv
uvicorn app.main:app --reload
```

Check http://localhost:8000/api/health — `yolo_mode` reads `"model"`
(`yolo_tier: "two_model"`) out of the box, since `ai/weights/` already
ships two real, separately-sourced models (stock COCO YOLOv8n for
vehicles + a community fire/smoke YOLOv8s, domain-transferred rather than
exhaust-specific — see `ai/weights/README.md`). It only falls back to
`"mock"` if `ultralytics` isn't installed, a weights file goes missing, or
`FORCE_MOCK_INFERENCE=1` is set. Once you train a unified model
(`ai/training/train.py`) and drop it in as `ai/weights/best.pt`, `yolo_tier`
switches to `"unified_model"` automatically — no code changes.

```bash
cd mobile
flutter pub get
flutter run --dart-define=API_BASE_URL=http://10.0.2.2:8000/api   # Android emulator
```

## Positive path (smoke detected)

1. Home → **Capture Vehicle**.
2. Take/select a photo showing a vehicle with a visible smoke plume.
3. AI Analysis screen shows vehicle %, smoke %, bounding boxes, and
   **"Reporting Eligible"**.
4. **Continue to License Plate** → capture the plate → OCR extracts (or
   asks you to type) the registration number → **Confirm & Continue**.
5. Grant location permission → screen shows a resolved place/landmark name
   (e.g. "Anna Salai, Chennai") and the timestamp → **Continue**.
6. Report Preview shows the full structured report ("Suspected visible
   exhaust smoke was observed…") → **Report on X** opens X with the text
   pre-filled and the authority handle included; you review and post it
   yourself.

## Negative path (no smoke)

1. Home → **Capture Vehicle**.
2. Photo of a vehicle with no visible smoke.
3. AI Analysis shows Vehicle ✓, Smoke "Not Detected", **"Reporting
   Disabled"**, and the required message: *"Visible exhaust smoke could not
   be detected. Reporting cannot continue."*
4. Only **Retake Photo** / **Cancel** are available — there is no path from
   here to the reporting screens.

## What to say if asked about accuracy

The vehicle and smoke detections in this build are real inference from real
model weights — not the OpenCV mock heuristic — but neither model was
trained on this app's own dataset. Vehicle detection is a stock,
well-validated COCO model; smoke detection is a community model
domain-transferred from general fire/smoke photos, not exhaust-pipe-specific
(see `ai/weights/README.md` for exactly where each came from and how each
was spot-checked). Report the numbers honestly as "real detection, unvalidated
on our specific use case" rather than as a benchmarked accuracy claim. The
`mode`/`yolo_tier` fields in every `/api/detect` response (and the AI
Analysis screen) always say which tier actually produced a given result —
`"mock"` only appears if inference genuinely fell back to the OpenCV
heuristic (see the Positive/Negative path steps above for what that looks
like). Training a unified model on your own dataset
(`ai/training/train.py` → `ai/weights/best.pt`) is what moves you from
"real but unvalidated" to a model you can report `precision`/`recall`/`mAP`
numbers for — see `ai/training/visualize_eval.py`.
