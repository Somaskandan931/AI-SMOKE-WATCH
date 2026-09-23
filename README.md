# SmokeWatch — Detect. Verify. Report.

**AI-assisted visible vehicle-exhaust smoke reporting for mobile.**

SmokeWatch is a Flutter + FastAPI application that helps a user document a vehicle that appears to be emitting visible exhaust smoke. The app deliberately separates **detection**, **verification**, and **reporting**:

1. Capture a vehicle image.
2. Run AI detection for a vehicle and visible smoke.
3. Require the smoke to be spatially associated with the detected vehicle.
4. Capture/verify the license plate.
5. Capture the user's location and timestamp.
6. Generate a structured **suspected-issue** report.
7. Let the user review it and open an X composer with the text pre-filled.

The system does **not** claim to measure emissions or establish a legal violation. It detects visible smoke in an image and produces evidence for human/authority review.

---

## Why this project exists

A simple "smoke / no smoke" classifier is not enough for this workflow.

A useful reporting pipeline needs to answer several separate questions:

- Is there a vehicle in the image?
- Is visible smoke present?
- Is that smoke actually associated with the vehicle rather than unrelated haze?
- What is the vehicle's registration number?
- Where and when was the evidence captured?
- Can the user review the generated report before sharing it?

SmokeWatch turns those questions into a single guided workflow.

---

## System overview

```mermaid
flowchart LR
    A["Flutter Mobile App"] -->|"Image upload"| B["FastAPI Backend"]

    B --> C["Vehicle Detector"]
    B --> D["Smoke Detector"]

    C --> E["Spatial Association"]
    D --> E

    E --> F{"Vehicle + smoke +\nconfidence + association?"}

    F -->|"No"| G["Reporting disabled"]
    F -->|"Yes"| H["License Plate Detector"]

    H --> I["OCR"]
    I --> J["User verifies / edits plate"]

    J --> K["Location + timestamp"]
    K --> L["Report Generator"]
    L --> M["Report Preview"]
    M --> N["X share-intent URL"]
    N --> O["User reviews and posts"]
```

The backend also supports a **unified YOLO model** when one is trained and placed in `ai/weights/best.pt`. Until then, the shipped configuration uses two separate real YOLO models: one for vehicles and one for smoke.

Rendered version of the same flow, including the explicit "User Confirms?" step before anything is posted to X:

![SmokeWatch system architecture and decision flow](./System_architecture.png)

---

## Competitive landscape

Existing civic-reporting apps — Bangalore's *Public Eye*, Delhi's *Traffic Prahari*, Goa's *Traffic Sentinel*, Chennai's *Namma Chennai*, and *CitizenLens* — all let a citizen submit a photo, video, and location, but each depends on the citizen alone to identify the violation before submitting. SmokeWatch's difference is the AI-verification step in the middle: the reporting workflow only unlocks once the model has actually detected smoke associated with a vehicle in the photo.

![Existing manual-reporting flow vs. SmokeWatch's AI-gated flow](./comparison.png)

| App | Scope | Gap vs. SmokeWatch |
|---|---|---|
| Public Eye (Bangalore Traffic Police) | General traffic violations | Manual citizen identification only, no AI verification |
| Traffic Prahari (Delhi Traffic Police) | General traffic violations, incl. reg. no./date/time/location | Same — no computer-vision gate before reporting |
| Traffic Sentinel (Goa Police) | General traffic violations | Same |
| Namma Chennai (Greater Chennai Corporation) | General civic grievances | Not violation- or vehicle-specific |
| CitizenLens | Civic issues, AI-assisted categorization, X/WhatsApp escalation | Categorizes after capture; doesn't gate reporting on a detection |

---

# The AI pipeline — what actually happens under the hood

The most important part of SmokeWatch is that the model is not simply looking at an image and returning one global `"smoke": true` answer.

For the YOLO detector, the image is transformed into feature maps at multiple spatial scales. The detection head makes predictions at thousands of locations, producing candidate bounding boxes and class scores. The application then turns those raw detections into the higher-level decision:

> **"Is there a vehicle with sufficiently confident visible smoke that is spatially associated with it?"**

Here is the pipeline tied directly to the implementation.

## 1. Backbone — feature extraction

For a typical `640 × 640` YOLO input, the backbone progressively reduces spatial resolution while increasing the number of feature channels.

Conceptually:

```text
640 × 640 × 3
      ↓
320 × 320
      ↓
160 × 160
      ↓
 80 × 80
      ↓
 40 × 40
      ↓
 20 × 20
```

The early convolutional/C2f stages learn lower-level visual features such as:

- edges
- corners
- textures
- color/brightness patterns

Deeper layers represent increasingly abstract patterns that can contribute to recognizing objects and smoke-like structures.

For the smoke model, this can mean features corresponding to shapes, textures, contrast and diffuse patterns associated with smoke. The model does **not** explicitly encode a rule such as "gray pixels above a tailpipe"; those visual representations are learned during training.

---

## 2. Neck — multi-scale feature fusion

The YOLO architecture then combines information from different resolutions.

Smoke can appear at very different apparent sizes:

- a small, distant plume
- a medium-sized exhaust cloud
- a large diffuse plume close to the camera

The feature pyramid therefore exposes information at multiple scales.

For a `640 × 640` input, the three principal detection resolutions correspond to:

```text
80 × 80 = 6,400 locations
40 × 40 = 1,600 locations
20 × 20 =   400 locations
                         ─────────
                         8,400 locations
```

This is why the detection head can work with both relatively small and relatively large objects.

---

## 3. Detection head — boxes + classes at every location

At each detection location, the YOLO head predicts information used to construct candidate bounding boxes and class scores.

For the YOLOv8-style head used here, bounding-box regression uses a **Distribution Focal Loss (DFL)** representation with `reg_max = 16`: each edge is represented through a distribution over 16 bins and decoded into box coordinates.

For the smoke model, the relevant class is:

```text
smoke
```

For the vehicle model, the implementation maps these COCO classes into the application's single logical label:

```text
car
truck
bus
motorcycle
        ↓
     vehicle
```

The application therefore receives detections in a common representation:

```text
label
confidence
(x1, y1, x2, y2)
```

For example:

```text
vehicle       0.94   (x1, y1, x2, y2)
exhaust_smoke 0.89   (x1, y1, x2, y2)
```

### Why are there thousands of predictions?

Because the model is evaluating many spatial positions across multiple feature-map resolutions.

Most candidate predictions are irrelevant. Ultralytics' inference pipeline applies confidence filtering and **Non-Maximum Suppression (NMS)** before the final detections are returned to `yolo_service.py`.

That is why several neighboring predictions around one object can eventually become one usable bounding box.

---

## 4. Two models currently work independently

The shipped real-inference configuration is a **two-model system**:

```text
                    ┌─────────────────────┐
640×640 image ────> │ Vehicle YOLOv8n     │ ──> vehicle boxes
                    └─────────────────────┘

                    ┌─────────────────────┐
640×640 image ────> │ Smoke YOLOv8s       │ ──> smoke boxes
                    └─────────────────────┘
```

Neither model needs to know about the other.

The vehicle model detects vehicles.

The smoke model detects smoke.

The application combines their results afterward.

This distinction is important: **the vehicle/smoke association logic is application code, not something the two currently-shipped models learn jointly.**

---

## 5. Confidence threshold and application decision

The backend configuration contains:

```text
SMOKE_CONFIDENCE_THRESHOLD = 0.5
```

A smoke detection below this threshold cannot unlock reporting.

However, this `0.5` value is specifically an **application reporting gate**. The smoke model's raw Ultralytics prediction call and its internal NMS have their own inference behavior before `yolo_service.py` receives the final detections.

After inference, the service selects the highest-confidence vehicle and highest-confidence smoke detection and evaluates the reporting conditions.

---

## 6. Spatial association — the important step after detection

A smoke detector finding smoke somewhere in a photograph does not automatically mean the nearby vehicle produced it.

SmokeWatch therefore performs a spatial association check.

Conceptually:

```text
             smoke
          ┌──────────┐
          │  plume   │
          └──────────┘
              ↓
       ┌─────────────────┐
       │ expanded        │
       │ vehicle region  │
       │                 │
       │     vehicle     │
       │    ┌───────┐    │
       │    │       │    │
       │    └───────┘    │
       └─────────────────┘
```

The vehicle bounding box is expanded to tolerate a plume extending above or behind the vehicle.

The implementation uses:

```text
horizontal padding = 25% of vehicle width
vertical padding   = 60% of vehicle height
```

The smoke box must overlap this expanded region.

This prevents a random hazy region elsewhere in the image from automatically enabling the reporting workflow.

---

## 7. Final reporting gate

The final decision is:

```text
vehicle_detected
        AND
smoke_detected
        AND
smoke_confidence >= 0.5
        AND
smoke_associated_with_vehicle
        ↓
reporting_allowed = true
```

Otherwise:

```text
reporting_allowed = false
```

The same requirement is enforced again by the report API, providing a second server-side check rather than trusting only the mobile UI.

---

# Model tiers

SmokeWatch supports three detection tiers.

## Tier 1 — Unified model

If either of these files exists:

```text
ai/weights/best.pt
ai/weights/last.pt
```

the backend attempts to load a single unified YOLO model first.

Expected classes:

```text
vehicle
exhaust_smoke
```

This is the architecture intended for a properly trained project-specific model.

### Training run on record

A `train.py` run has actually been completed against `ai/configs/smoke.yaml` (see [`ai/weights/README.md`](ai/weights/README.md) for the exact tier-priority logic). Ultralytics wrote its usual run artifacts to `ai/runs/detect/train/` and a separate validation pass to `ai/runs/detect/val/`:

```text
ai/runs/detect/train/
├── weights/                     # best.pt / last.pt for this run
├── args.yaml                    # exact hyperparameters used
├── results.csv                  # per-epoch loss/precision/recall/mAP
├── results.png                  # results.csv plotted across all epochs
├── confusion_matrix.png
├── confusion_matrix_normalized.png
├── BoxP_curve.png / BoxR_curve.png / BoxF1_curve.png / BoxPR_curve.png
├── labels.jpg                   # dataset label distribution
├── train_batch*.jpg             # sample training batches with GT boxes
└── val_batch*_labels.jpg / val_batch*_pred.jpg   # GT vs. prediction, per batch

ai/runs/detect/val/               # standalone validate.py pass over the same weights
├── BoxP_curve.png / BoxR_curve.png / BoxF1_curve.png / BoxPR_curve.png
├── confusion_matrix.png / confusion_matrix_normalized.png
└── val_batch*_labels.jpg / val_batch*_pred.jpg
```

**`ai/weights/best.pt` and the raw dataset under `ai/dataset/` are intentionally not committed to this repository** — the trained checkpoint and image data made the repo too large to keep in version control. `ai/runs/` is kept because the plots and `results.csv` are small and are the actual evidence of what this run measured; regenerate the weights by re-running `train.py` against a re-downloaded dataset (see "Training a project-specific model" below) if you need `best.pt` back locally. Read the numbers in `results.csv` / `results.png` yourself before quoting a precision, recall, or mAP figure anywhere — none are asserted in this README.

## Tier 2 — Current shipped two-model setup

If a unified model is unavailable, the backend automatically looks for:

```text
ai/weights/vehicle_yolov8n.pt
ai/weights/smoke_yolov8s.pt
```

These are separate real YOLO models:

- **Vehicle:** stock Ultralytics YOLOv8n trained on COCO; `car`, `truck`, `bus`, and `motorcycle` are mapped to the application's `vehicle` label.
- **Smoke:** a community YOLOv8s model trained on general fire/smoke imagery. It is **not exhaust-pipe-specific**, so its output should be treated as real inference but not as a validated exhaust-emission benchmark.

## Tier 3 — Mock fallback

If real model inference cannot load, the backend falls back to an OpenCV heuristic.

It is intentionally marked:

```json
{
  "mode": "mock"
}
```

It is not presented as trained-model inference.

This makes the application demonstrable even when large ML dependencies or weights are unavailable.

---

# Plate detection and OCR

Smoke detection is only one part of the workflow.

Once reporting is allowed:

```text
Vehicle image
     ↓
License plate detector
     ↓
Plate crop
     ↓
OCR
     ↓
User verification/edit
```

The plate detector uses:

```text
ai/weights/plate_best.pt
```

when available and falls back to an OpenCV contour/aspect-ratio heuristic if necessary.

OCR supports:

1. EasyOCR
2. PaddleOCR
3. Manual entry fallback

The user can verify or edit the OCR result before continuing.

---

# Report generation

The backend deliberately generates a **suspected issue** report rather than declaring a legal violation.

The generated report includes:

- registration number
- location
- date/time
- visible-smoke observation
- authority handle
- X share-intent URL

The X flow is intentionally:

```text
Backend
  ↓
Generate text
  ↓
Create X intent URL
  ↓
Open X composer
  ↓
User reviews
  ↓
User posts
```

The application does **not** silently publish a report.

---

# Product workflow

## Positive path

```text
Home
  ↓
Capture Vehicle
  ↓
AI Analysis
  ↓
Vehicle + smoke + spatial association
  ↓
License Plate
  ↓
OCR verification
  ↓
Location + timestamp
  ↓
Report Preview
  ↓
Open X composer
  ↓
User confirms/post
```

## Negative path

If the image does not satisfy the detection gate:

```text
Capture Vehicle
      ↓
AI Analysis
      ↓
No qualifying smoke
      ↓
Reporting Disabled
      ↓
Retake / Cancel
```

The user cannot bypass the smoke verification step and jump directly into the reporting flow.

---

# Key features

| Feature | Implementation |
|---|---|
| Vehicle detection | Ultralytics YOLOv8n / unified YOLO |
| Visible smoke detection | YOLOv8s / unified YOLO |
| Multi-scale object detection | YOLO feature pyramid + detection head |
| Vehicle/smoke association | Bounding-box spatial gating |
| Confidence gate | Configurable smoke confidence threshold |
| License plate detection | YOLO plate detector + OpenCV fallback |
| Plate OCR | EasyOCR / PaddleOCR / manual fallback |
| Location | Flutter `geolocator` |
| Reverse geocoding | OpenStreetMap Nominatim |
| Report generation | FastAPI service |
| Social sharing | X share-intent URL |
| User confirmation | Report preview before opening X |
| Backend | FastAPI |
| Mobile | Flutter |
| Training | Ultralytics + Roboflow dataset workflow |
| Persistence | PostgreSQL schema provided, optional/not currently wired |
| Containerization | Docker Compose |

---

# Tech stack

### Mobile

- Flutter
- Dart
- `image_picker`
- `geolocator`
- `url_launcher`
- Provider-based state/model flow

### Backend

- Python
- FastAPI
- Pydantic
- OpenCV
- NumPy
- Pillow
- Ultralytics
- pytest

### AI / training

- YOLOv8 / Ultralytics
- Roboflow dataset export
- OpenCV
- Matplotlib for evaluation visualizations

### Infrastructure

- Docker / Docker Compose
- PostgreSQL schema
- Optional X API integration code

---

# Project structure

```text
.
├── ai/
│   ├── configs/
│   │   ├── smoke.yaml
│   │   └── smoke_only.yaml
│   ├── dataset/
│   │   ├── raw/
│   │   ├── processed/
│   │   └── download_from_roboflow.py
│   ├── training/
│   │   ├── train.py
│   │   ├── validate.py
│   │   └── visualize_eval.py
│   └── weights/
│       ├── vehicle_yolov8n.pt
│       ├── smoke_yolov8s.pt
│       ├── plate_best.pt
│       └── best.pt              # optional future unified model
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── detection.py
│   │   │   ├── plate.py
│   │   │   └── reports.py
│   │   ├── services/
│   │   │   ├── yolo_service.py
│   │   │   ├── plate_detector.py
│   │   │   ├── ocr_service.py
│   │   │   └── report_service.py
│   │   ├── config.py
│   │   └── main.py
│   ├── tests/
│   ├── Dockerfile
│   └── requirements.txt
│
├── database/
│   └── schema.sql
│
├── docs/
│   ├── architecture.md
│   ├── api.md
│   └── demo.md
│
├── mobile/
│   ├── lib/
│   │   ├── screens/
│   │   ├── services/
│   │   ├── models/
│   │   └── widgets/
│   ├── pubspec.yaml
│   └── NATIVE_SETUP.md
│
├── docker-compose.yml
├── .env.example
├── CHANGES.md
├── STATUS.md
└── README.md
```

---

# Quick start

## 1. Backend

From the repository root:

```bash
cd backend
python -m venv .venv
```

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Linux / macOS

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Create the backend environment file:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Start FastAPI:

```bash
uvicorn app.main:app --reload
```

Open:

- Swagger UI: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/api/health`

A healthy real-inference setup should report a model tier such as:

```json
{
  "status": "ok",
  "yolo_mode": "model",
  "yolo_tier": "two_model",
  "plate_mode": "model"
}
```

The exact response depends on which weights and dependencies are available.

---

# 2. Install OCR support

EasyOCR is optional because the backend can fall back to manual plate entry.

For real OCR:

```bash
pip install easyocr
```

The OCR service also supports PaddleOCR when installed.

---

# 3. Run the test suite

From `backend/`:

```bash
pytest
```

The repository currently contains the backend test suite and is designed to use deterministic mock inference during tests so that tests do not depend on large model downloads or non-deterministic ML output.

For the latest repository-specific verification details, see [`STATUS.md`](STATUS.md).

---

# 4. Run the Flutter app

```bash
cd mobile
flutter pub get
```

Run against the backend.

### Android emulator

```bash
flutter run --dart-define=API_BASE_URL=http://10.0.2.2:8000/api
```

### Physical device

Use the host machine's LAN IP instead of `10.0.2.2`, for example:

```bash
flutter run --dart-define=API_BASE_URL=http://192.168.1.10:8000/api
```

The phone and development machine must be able to reach each other over the network.

See [`mobile/NATIVE_SETUP.md`](mobile/NATIVE_SETUP.md) for native camera/location permission setup.

---

# Training a project-specific model

The current shipped smoke model is useful for demonstrating the pipeline, but it is **not trained specifically for vehicle exhaust**.

The intended research/development path is to train on a dataset containing:

### Positive examples

- visible exhaust smoke
- different smoke densities
- different vehicle types
- day/night conditions
- different camera distances
- different camera angles
- different backgrounds

### Difficult negatives

- fog
- dust
- steam
- road haze
- clouds
- shadows
- motion blur
- reflections
- other gray/white regions

The dataset configuration for the unified model is:

```text
ai/configs/smoke.yaml
```

with:

```yaml
names:
  0: vehicle
  1: exhaust_smoke
```

## Download the dataset

Create the root `.env`:

```bash
cp .env.example .env
```

Set:

```text
ROBOFLOW_API_KEY=your_key_here
```

Then:

```bash
cd ai
pip install -r requirements.txt

cd dataset
python download_from_roboflow.py --sources sources.yaml
```

## Train

```bash
cd ../training

python train.py \
  --data ../configs/smoke.yaml \
  --model yolov8n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16
```

Ultralytics will produce:

```text
runs/detect/train/weights/best.pt
```

Copy the trained model to:

```text
ai/weights/best.pt
```

On the next backend startup, that unified model takes priority over the separate vehicle and smoke models.

---

# Model evaluation

Validation reports include:

- Precision
- Recall
- mAP@50
- mAP@50-95

Run:

```bash
cd ai/training
python validate.py \
  --weights ../weights/best.pt \
  --data ../configs/smoke.yaml
```

For plots and a structured metrics report:

```bash
python visualize_eval.py \
  --weights ../weights/best.pt \
  --data ../configs/smoke.yaml
```

Expected output includes:

```text
ai/training/results/
├── metrics.json
├── metrics.md
├── confusion_matrix.png
├── PR_curve.png
├── F1_curve.png
├── P_curve.png
├── R_curve.png
└── per_class_metrics.png
```

Do not describe a model as "accurate" from a single demo image. Use the validation-set metrics and clearly state the dataset and evaluation conditions.

The one training run already on record used Ultralytics' own default output layout (`ai/runs/detect/train/`, `ai/runs/detect/val/`) rather than this custom `ai/training/results/` layout — see "Training run on record" under Tier 1 above for what's actually in there.

---

# API

The backend exposes the following core endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | Backend/model health |
| `POST` | `/api/detect` | Vehicle + smoke detection |
| `POST` | `/api/plate/detect` | License plate localization |
| `POST` | `/api/plate/ocr` | Plate OCR |
| `POST` | `/api/report/generate` | Generate structured report + X intent URL |

Full request/response details are in [`docs/api.md`](docs/api.md).

Interactive API documentation is available at:

```text
http://localhost:8000/docs
```

---

# Example detection decision

A successful detection can conceptually look like:

```json
{
  "vehicle_detected": true,
  "smoke_detected": true,
  "vehicle_confidence": 0.94,
  "smoke_confidence": 0.89,
  "smoke_associated_with_vehicle": true,
  "reporting_allowed": true,
  "mode": "model"
}
```

The important field is not simply `smoke_detected`.

The reporting decision requires the complete condition:

```text
vehicle
+ smoke
+ sufficient smoke confidence
+ spatial association
= reporting allowed
```

---

# Screenshots

The repository can include screenshots under:

```text
screenshots/
```

Recommended captures:

| Screen | Filename |
|---|---|
| Home | `01_home.png` |
| Vehicle capture | `02_capture.png` |
| Positive AI analysis | `03_analysis_smoke_detected.png` |
| Negative AI analysis | `04_analysis_no_smoke.png` |
| Plate capture | `05_plate_capture.png` |
| OCR verification | `06_ocr_verify.png` |
| Location | `07_location.png` |
| Report preview | `08_report_preview.png` |
| X composer | `09_x_intent.png` |

Once real screenshots are added, they can be embedded directly here to turn the repository into a visual project demo.

---

# Engineering decisions

## Why detection is separated from reporting

The neural network should answer visual questions. It should not decide whether the user should publish a civic report.

That separation makes the system easier to reason about:

```text
ML inference
    ↓
structured detections
    ↓
application rules
    ↓
reporting eligibility
    ↓
human review
```

This also means the smoke model can be replaced without rewriting the reporting workflow.

## Why there is a mock mode

The project must remain runnable when:

- YOLO dependencies are not installed
- model weights are unavailable
- tests need deterministic behavior
- a lightweight development environment is being used

Mock mode is therefore an explicit fallback, not an attempt to disguise heuristic output as model inference.

## Why the user reviews the report

The generated report is an assistive output.

The user sees the final content before it is handed to the X composer. The backend does not silently publish on the user's behalf.

---

# Current status

| Component | State |
|---|---|
| FastAPI backend | Implemented |
| Vehicle detection | Real YOLO model + fallback |
| Smoke detection | Real YOLO model + fallback |
| Vehicle/smoke spatial association | Implemented |
| Reporting gate | Implemented |
| Plate detection | Real YOLO model + fallback |
| OCR | EasyOCR / PaddleOCR / manual fallback |
| Report generation | Implemented |
| X share intent | Implemented |
| Flutter workflow | Code-complete; requires local Flutter build/run verification |
| Unified project-specific YOLO | One training run completed (`ai/runs/detect/train/`, `.../val/`); `best.pt` and the dataset are not committed to the repo |
| PostgreSQL | Schema provided; not currently wired into API |
| Docker | Configuration provided |
| Automated backend tests | Included |

See [`STATUS.md`](STATUS.md) and [`CHANGES.md`](CHANGES.md) for repository-specific verification and recent fixes.

---

# Known limitations

### 1. The shipped smoke model is not exhaust-specific

The current *shipped* (Tier 2) smoke detector comes from general fire/smoke imagery. It is therefore a **domain-transfer baseline**, not evidence of production-grade vehicle-exhaust detection.

A project-specific training run has been completed once (see "Training run on record" under Model tiers), but its `best.pt` isn't committed to this repo, and its metrics haven't been asserted here — read `ai/runs/detect/train/results.csv` and the confusion matrix yourself before citing a number.

### 2. Visible smoke is not an emissions measurement

SmokeWatch does not measure:

- CO
- CO₂
- NOx
- particulate concentration
- opacity using a calibrated instrument
- legal emissions compliance

It only analyzes visible smoke in an image.

### 3. Detection is not proof of a legal violation

A generated report uses wording such as:

> "Suspected visible exhaust smoke was observed from a vehicle."

The system intentionally does not convert an image detection into a legal conclusion.

### 4. Real-world conditions can produce false positives or false negatives

Examples include:

- fog
- dust
- steam
- lighting changes
- compression artifacts
- motion blur
- occlusion
- unusual vehicle shapes
- very faint smoke
- smoke outside the model's learned distribution

This is why difficult-negative data and validation metrics matter.

### 5. Flutter still requires device/emulator validation

The mobile code needs to be built and exercised in a Flutter environment with the appropriate native permissions and device/emulator configuration.

---

# Documentation

- [`docs/architecture.md`](docs/architecture.md) — system architecture and data flow
- [`docs/api.md`](docs/api.md) — API reference
- [`docs/demo.md`](docs/demo.md) — end-to-end demo procedure
- [`mobile/NATIVE_SETUP.md`](mobile/NATIVE_SETUP.md) — Flutter native setup
- [`ai/weights/README.md`](ai/weights/README.md) — model provenance and weight tiers

---

# License

Add the project's intended license here before publishing the repository publicly.

If model weights or datasets are redistributed, also verify and document the licenses and attribution requirements of those individual assets.

---

## One-line summary

**SmokeWatch combines YOLO-based vehicle/smoke detection, spatial verification, license-plate OCR, location capture, and human-reviewed report generation into one end-to-end mobile workflow for documenting suspected visible vehicle exhaust smoke.**