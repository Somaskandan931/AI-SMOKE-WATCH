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

> **Repository note:** The README documents a completed model benchmark and
> its local evaluation artifacts, but the reduced repository intentionally
> excludes the original dataset and large trained checkpoints. Benchmark
> results (figures, tables, JSON) live under `ai/benchmark/results/` and are
> included. Paths under `ai/runs/detect/` refer to an earlier, superseded
> training run and the original local machine unless those artifacts are
> copied into the repository.

> **Status:** a leakage-free five-model benchmark has been run (see
> [Benchmark results](#benchmark-results-leakage-free-split)). The deployed
> app has **not** yet been switched to a benchmark-selected detector.

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

> **Note from the benchmark:** the validation-frozen operating thresholds for
> the benchmarked smoke-only detectors were 0.2–0.3 for YOLO models and 0.9 for
> Faster R-CNN (see [Benchmark results](#benchmark-results-leakage-free-split)),
> not 0.5. The 0.5 reporting gate was not tuned on this benchmark; revisit it
> once a detector is selected.

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

How this rule compares with IoU, centroid and directional alternatives on a small labelled set is reported in [Vehicle–smoke association](#vehiclesmoke-association).

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

## Tier 1 — Unified project-specific model

If a trained unified checkpoint is available at:

```text
ai/weights/best.pt
```

or:

```text
ai/weights/last.pt
```

the backend can use it as the highest-priority YOLO detector.

Expected classes:

```text
vehicle
exhaust_smoke
```

The training configuration is:

```text
ai/configs/smoke.yaml
```

with:

```yaml
names:
  0: vehicle
  1: exhaust_smoke
```

### Earlier training run (superseded)

An earlier project-specific Ultralytics training run was completed and produced the standard run artifacts under:

```text
ai/runs/detect/train/
```

A separate validation pass produced:

```text
ai/runs/detect/val/
```

The original local Windows artifact locations are:

```text
D:\PycharmProjects\AI_Smoke_Watch\ai\runs\detect\train\
D:\PycharmProjects\AI_Smoke_Watch\ai\runs\detect\val\
```

> **This run predates the leakage-free split.** The original dataset export
> contained several augmented copies of the same photo in different splits
> (59 of 99 test images shared a source photo with train/val — see
> [Dataset and leakage audit](#dataset-and-leakage-audit)). Metrics from this
> run are therefore optimistic and **must not be cited as benchmark results**.
> Use the numbers in the benchmark section instead. The artifacts below are kept
> for provenance and for the visual examples only.

The most important artifacts of that run are:

| Artifact | Purpose |
|---|---|
| `ai/runs/detect/train/args.yaml` | Exact training configuration and hyperparameters recorded by Ultralytics |
| `ai/runs/detect/train/results.csv` | Per-epoch training/validation losses and detection metrics |
| `ai/runs/detect/train/results.png` | Training history plotted from `results.csv` |
| `ai/runs/detect/train/confusion_matrix.png` | Raw validation confusion matrix |
| `ai/runs/detect/train/confusion_matrix_normalized.png` | Normalized confusion matrix |
| `ai/runs/detect/train/BoxP_curve.png` | Precision curve |
| `ai/runs/detect/train/BoxR_curve.png` | Recall curve |
| `ai/runs/detect/train/BoxF1_curve.png` | F1-score curve |
| `ai/runs/detect/train/BoxPR_curve.png` | Precision-recall curve |
| `ai/runs/detect/train/labels.jpg` | Dataset label-distribution visualization |
| `ai/runs/detect/train/val_batch{0,1,2}_pred.jpg` | Example validation predictions |
| `ai/runs/detect/train/val_batch{0,1,2}_labels.jpg` | Ground-truth validation labels |
| `ai/runs/detect/train/train_batch{0,1,2}.jpg` | Training-batch visualization |
| `ai/runs/detect/val/*` | Standalone validation pass (confusion matrices, P/R/F1/PR curves, prediction samples) |

The `weights/` directory from that training run contains the trained checkpoints (`best.pt` / `last.pt`) when the run is retained locally. Those large checkpoint files are intentionally **not included in the reduced repository**.

Likewise, the training dataset is intentionally **not included**. The repository retains the training configuration and evaluation evidence, while the image dataset and large trained checkpoints can be restored/recreated when needed.

### Run-artifact interpretation

The artifacts answer different questions:

```text
args.yaml
   ↓
"What configuration produced this run?"

results.csv / results.png
   ↓
"How did training and validation metrics change over epochs?"

confusion_matrix*.png
   ↓
"Which classes were confused with each other?"

BoxP / BoxR / BoxF1 / BoxPR
   ↓
"How do precision, recall, F1 and precision-recall behavior change
 across confidence thresholds?"

val_batch*_labels.jpg
   ↓
"What was actually annotated in the validation examples?"

val_batch*_pred.jpg
   ↓
"What did the trained detector predict on those examples?"
```

## Tier 2 — Two-model setup

The backend supports a two-model configuration using:

```text
ai/weights/vehicle_yolov8n.pt
ai/weights/smoke_yolov8s.pt
```

These are separate real YOLO models:

- **Vehicle:** stock Ultralytics YOLOv8n trained on COCO; `car`, `truck`, `bus`, and `motorcycle` are mapped to the application's `vehicle` label.
- **Smoke:** a community YOLOv8s model trained on general fire/smoke imagery. It is **not exhaust-pipe-specific**, so its output should be treated as a domain-transfer baseline rather than a validated exhaust-emission benchmark.

**Repository note:** these weight files are intentionally absent from the supplied reduced ZIP. `ai/weights/` currently contains the model-tier documentation only. Restore the required weights locally if you want to run the real two-model tier.


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
| Detector benchmark | YOLOv8n/s, YOLO11n/s, Faster R-CNN on a leakage-free split |
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

- YOLOv8 / YOLO11 / Ultralytics
- torchvision Faster R-CNN (MobileNetV3-FPN)
- Roboflow dataset export
- OpenCV
- Matplotlib for evaluation visualizations

### Infrastructure

- Docker / Docker Compose
- PostgreSQL schema
- Optional X API integration code

---

# Project structure

The project is organized into four main areas: AI/training, FastAPI backend,
Flutter mobile, and supporting documentation/infrastructure.

```text
.
├── ai/
│   ├── benchmark/
│   │   ├── README.md               # run order for the benchmark
│   │   └── results/                # metrics, tables, figures, association/hard-case outputs
│   │       ├── figures/            # fig2–fig6
│   │       ├── association_vis/    # vehicle–smoke association overlays
│   │       ├── preds/              # raw per-model predictions (val/test)
│   │       ├── tables.md
│   │       ├── test_metrics.json
│   │       ├── thresholds.json
│   │       ├── efficiency.json / efficiency_cpu.json
│   │       ├── association_summary.json / association_candidates.csv
│   │       ├── hard_cases_summary.json / hard_cases_detail.csv
│   │       ├── split_audit.json
│   │       └── reproducibility.json
│   ├── configs/
│   │   ├── smoke.yaml
│   │   ├── smoke_only.yaml
│   │   └── benchmark.yaml
│   ├── notebooks/
│   ├── training/
│   │   ├── train.py
│   │   ├── validate.py
│   │   ├── evaluate.py
│   │   ├── visualize_eval.py
│   │   └── yolov8n.pt              # present in supplied ZIP snapshot
│   └── weights/
│       └── README.md               # model-tier/provenance documentation
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── detection.py
│   │   │   ├── plate.py
│   │   │   └── reports.py
│   │   ├── models/
│   │   │   ├── plate_detector.py
│   │   │   └── smoke_detector.py
│   │   ├── schemas/
│   │   │   ├── detection_schema.py
│   │   │   └── report_schema.py
│   │   ├── services/
│   │   │   ├── yolo_service.py
│   │   │   ├── plate_detector.py
│   │   │   ├── ocr_service.py
│   │   │   ├── report_service.py
│   │   │   └── x_client.py
│   │   ├── utils/
│   │   ├── config.py
│   │   └── main.py
│   ├── tests/
│   ├── Dockerfile
│   └── requirements.txt
│
├── database/
│   ├── migrations/
│   └── schema.sql
│
├── docs/
│   ├── architecture.md
│   ├── api.md
│   └── demo.md
│
├── mobile/
│   └── android/                    # Android project scaffold in supplied ZIP
│
├── paper/
│   ├── paper_draft.md              # manuscript skeleton (numbers filled only from results/)
│   ├── figures/
│   └── tables/
│
├── docker-compose.yml
├── .env.example
└── README.md
```

### Reduced-repository note

The supplied ZIP intentionally omits the large dataset and the trained
`ai/weights/*.pt` checkpoints. It also contains only the Android portion of
the Flutter project in the supplied snapshot; the Flutter `lib/` source and
`pubspec.yaml` are not present in this ZIP snapshot.

If your Git working tree contains the Flutter source separately, the intended
mobile layout is:

```text
mobile/
├── lib/
│   ├── screens/
│   ├── services/
│   ├── models/
│   ├── widgets/
│   └── core/
├── pubspec.yaml
└── android/
```

This README therefore documents the application architecture while explicitly
avoiding the claim that omitted files are present in the reduced archive.

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

For Android native configuration, use the files under `mobile/android/` and verify camera/location permissions before running on a device.

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

The reduced repository does not include `ai/dataset/`. To retrain, restore or
recreate the dataset export and the dataset-download helper before running the
training pipeline.

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

The visualization script can generate a structured evaluation directory such
as:

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

For the multi-model comparison, use the benchmark protocol (`ai/configs/benchmark.yaml`, run order in `ai/benchmark/README.md`) rather than the single-model `validate.py` flow; its outputs are summarised below.

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

# Model results and project visuals

## System architecture

![SmokeWatch system architecture and decision flow](./System_architecture.png)

The architecture diagram shows the complete application flow from image capture
through AI detection, spatial association, plate verification, location capture,
report generation, and user confirmation.

## Product / workflow comparison

![Existing manual-reporting flow vs. SmokeWatch's AI-gated flow](./comparison.png)

This diagram illustrates the distinction between a conventional manual
reporting workflow and SmokeWatch's AI-gated workflow.

---

# Benchmark results (leakage-free split)

`ai/benchmark/` holds a reproducible comparison of **YOLOv8n, YOLOv8s, YOLO11n, YOLO11s and Faster R-CNN (MobileNetV3-FPN)** for smoke detection. Protocol: [`ai/configs/benchmark.yaml`](ai/configs/benchmark.yaml). Run order: [`ai/benchmark/README.md`](ai/benchmark/README.md). Every number below is copied from [`ai/benchmark/results/`](ai/benchmark/results/) (primarily [`tables.md`](ai/benchmark/results/tables.md)).

> **Scope.** The benchmark is *smoke-only* box detection (single class). Vehicle
> detection, plate OCR and the app flow are not part of these metrics. Results
> are from **one training seed per model**; confidence intervals reflect
> test-set sampling, not training variance.

## Dataset and leakage audit

The original Roboflow export contained several augmented copies of the same photo in different splits. A perceptual-hash grouped split (`hash_dist = 5`, seed 0) was built so that every source group lives in exactly one split.

| Split | Images | Smoke instances |
|---|---:|---:|
| train | 728 | 816 |
| val | 195 | 260 |
| test | 103 | 121 |
| **Total** | **1026** | **1197** |

| | Original split | Grouped split |
|---|---:|---:|
| Source groups | 558 | 558 |
| Groups spanning more than one split | 77 | **0** |
| Test images sharing a source with train/val | 59 of 99 | **0 of 103** |

Split manifest SHA-256: `b0b5c127d32d7cf79a27de557b550508be7d99af485450103c4a8a9604cd38d2` (see [`reproducibility.json`](ai/benchmark/results/reproducibility.json)).

## Training protocol

All YOLO models: 640 px, 50 epochs, batch 8, seed 0, optimizer `auto`, `lr0` 0.01, weight decay 5e-4, AMP, deterministic. Faster R-CNN (MobileNetV3-FPN, COCO-pretrained): 50 epochs, batch 8, SGD (momentum 0.9, wd 5e-4, lr 0.01), multi-step schedule with linear warm-up, horizontal flip + brightness/contrast jitter (no mosaic) — a budget-matched but not identical recipe because of framework differences.

Confidence thresholds were chosen on the **validation** set (maximum F1 over a 0.1–0.9 grid), **frozen**, then applied once to the test set. Metrics use IoU 0.5 for P/R/F1; 95% CIs are bootstrap intervals over test images. Own-evaluator mAP agrees with Ultralytics' built-in validator to within about 0.02 on the YOLO models (cross-checks in `tables.md`).

## Held-out test results

Test set: 103 images, 121 smoke instances.

| Model | Conf thr | Precision | Recall | F1 | mAP@50 [95% CI] | mAP@50-95 [95% CI] |
|---|---:|---:|---:|---:|---|---|
| YOLOv8n | 0.3 | 0.684 | 0.661 | 0.672 | 0.700 [0.601, 0.806] | 0.368 [0.308, 0.441] |
| YOLOv8s | 0.2 | 0.588 | 0.744 | 0.657 | 0.707 [0.610, 0.807] | 0.340 [0.281, 0.414] |
| YOLO11n | 0.3 | 0.740 | 0.636 | 0.684 | 0.720 [0.617, 0.824] | 0.364 [0.300, 0.435] |
| YOLO11s | 0.2 | 0.680 | 0.702 | 0.691 | 0.738 [0.636, 0.837] | 0.371 [0.311, 0.441] |
| Faster R-CNN (MobileNetV3-FPN) | 0.9 | 0.810 | 0.669 | 0.733 | 0.751 [0.666, 0.844] | 0.402 [0.342, 0.478] |

![Test mAP@50 and mAP@50-95 per model](./ai/benchmark/results/figures/fig2_benchmark.png)

*Fig. 2 — Test mAP per model.*

![Precision-recall curves on the test set](./ai/benchmark/results/figures/fig3_pr_curves.png)

*Fig. 3 — Precision–recall curves (IoU 0.5) on the test set.*

**Reading the table.** Faster R-CNN has the highest point estimates on every aggregate metric, but the 95% intervals of all five models overlap heavily (for example mAP@50 0.700 [0.601, 0.806] for YOLOv8n vs 0.751 [0.666, 0.844] for Faster R-CNN). With one seed and 103 test images the benchmark does **not** support a statistically reliable accuracy ranking; it supports "all five are in the same range".

## Threshold selection (validation)

Operating thresholds were picked on validation data only. The dashed line marks the frozen value.

![Precision, recall and F1 vs confidence threshold on validation](./ai/benchmark/results/figures/fig4_threshold_analysis.png)

*Fig. 4 — Validation precision/recall/F1 vs confidence threshold.*

| Model | Frozen thr | Val F1 at thr | Val mAP@50 | Val mAP@50-95 |
|---|---:|---:|---:|---:|
| YOLOv8n | 0.3 | 0.448 | 0.382 | 0.168 |
| YOLOv8s | 0.2 | 0.410 | 0.369 | 0.160 |
| YOLO11n | 0.3 | 0.417 | 0.366 | 0.173 |
| YOLO11s | 0.2 | 0.421 | 0.370 | 0.168 |
| Faster R-CNN (MobileNetV3-FPN) | 0.9 | 0.466 | 0.417 | 0.203 |

Two things to keep in mind: Faster R-CNN's frozen threshold sits at the edge of the searched grid and its F1 curve is nearly flat across thresholds (its scores are heavily saturated), so that value is a weak operating point rather than a sharply tuned one; and validation scores are far lower than test scores for every model (see [Limitations](#known-limitations)).

## Efficiency

Batch size 1, 640 px, in-memory BGR image, **including pre-/post-processing and NMS**. CPU: AMD64 (16 logical cores), PyTorch threads set to 4. GPU: CUDA device (PyTorch 2.5.1+cu121, Ultralytics 8.4.158, Windows 11). 309 timed runs per model. Raw files: [`efficiency_cpu.json`](ai/benchmark/results/efficiency_cpu.json), [`efficiency.json`](ai/benchmark/results/efficiency.json).

| Model | Params (M) | GFLOPs | File (MB) | CPU mean ms | CPU median ms | CPU P95 ms | GPU mean ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8n | 3.01 | 8.2 | 6.2 | 51.7 | 50.8 | 57.1 | 20.5 |
| YOLOv8s | 11.14 | 28.6 | 22.5 | 127.5 | 122.3 | 157.5 | 27.3 |
| YOLO11n | 2.59 | 6.5 | 5.5 | 57.0 | 55.3 | 67.2 | 23.9 |
| YOLO11s | 9.43 | 21.7 | 19.2 | 121.5 | 118.9 | 136.9 | 26.1 |
| Faster R-CNN (MobileNetV3-FPN) | 18.95 | – | 76.0 | 128.7 | 124.1 | 159.7 | 29.9 |

![Accuracy vs CPU latency](./ai/benchmark/results/figures/fig5_accuracy_latency.png)

*Fig. 5 — Test mAP@50-95 vs mean CPU latency.*

**Pareto-optimal (mAP@50-95 vs mean CPU latency): YOLOv8n, YOLO11s, Faster R-CNN (MobileNetV3-FPN).** In practical terms: YOLOv8n is the fastest and smallest of the three (about 52 ms, 6.2 MB) at an mAP@50-95 point estimate of 0.368; Faster R-CNN has the highest point estimate (0.402) at roughly 2.5× the CPU latency and 12× the file size. **No detector has been selected for deployment yet**, and the backend still ships the Tier 2 configuration.

## Qualitative examples

![Faster R-CNN qualitative examples on the test set](./ai/benchmark/results/figures/fig6_qualitative.png)

*Fig. 6 — Faster R-CNN at its frozen threshold; green = ground truth, red = prediction. Panels: best-IoU hit, a missed diffuse plume, an extra/misaligned box, and the smallest smoke instance.*

## Difficult cases

A small hand-assembled set of 10 images (6 with smoke, 4 verified negatives) in `photos for test/`, evaluated at image level with the frozen thresholds. **n is tiny — treat these as indicative only.**

| Model | Smoke images | Recall | Negative images | False-positive rate |
|---|---:|---:|---:|---:|
| YOLOv8n | 6 | 0.500 | 4 | 0.750 |
| YOLOv8s | 6 | 0.833 | 4 | 0.750 |
| YOLO11n | 6 | 0.333 | 4 | 0.000 |
| YOLO11s | 6 | 0.833 | 4 | 0.500 |
| Faster R-CNN (MobileNetV3-FPN) | 6 | 0.500 | 4 | 0.500 |

Images flagged as smoke per category (detected / total). For the `fog_mist`, `no_smoke` and `non_vehicle_smoke` rows, any detection is a false positive.

| Category | YOLOv8n | YOLOv8s | YOLO11n | YOLO11s | Faster R-CNN |
|---|---:|---:|---:|---:|---:|
| smoke | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| heavy_smoke | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| light_smoke | 0/2 | 1/2 | 0/2 | 1/2 | 1/2 |
| multi_vehicle | 1/2 | 2/2 | 0/2 | 2/2 | 0/2 |
| fog_mist | 1/1 | 1/1 | 0/1 | 0/1 | 1/1 |
| non_vehicle_smoke | 2/2 | 2/2 | 0/2 | 2/2 | 1/2 |
| no_smoke | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 |

The pattern is the one the dataset caveats predict: every model finds heavy smoke, light smoke is the weak spot, and fog or smoke from non-vehicle sources (a smoking bin, for example) produces false positives. YOLO11n has no false positives but also the lowest recall, which is a threshold trade-off rather than evidence of better discrimination. Per-image detail: [`hard_cases_detail.csv`](ai/benchmark/results/hard_cases_detail.csv); box-level per-category results are in `tables.md`.

## Vehicle–smoke association

Four association rules were compared on **16 labelled vehicle–smoke pairs (13 positive, 3 negative)** built from detector outputs on 12 images ([`association_summary.json`](ai/benchmark/results/association_summary.json), [`association_candidates.csv`](ai/benchmark/results/association_candidates.csv)).

| Rule | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|
| IoU overlap | 1.000 | 0.769 | 0.870 | 0.812 |
| Smoke-centroid inside vehicle box | 1.000 | 0.923 | 0.960 | 0.938 |
| **Padded (deployed: +25% width, +60% height)** | 0.929 | 1.000 | 0.963 | 0.938 |
| Directional | 1.000 | 0.846 | 0.917 | 0.875 |

![Association overlays (green: vehicle, red: smoke)](./ai/benchmark/results/association_overview.jpg)

*Association overlays for all 12 images (individual files in [`ai/benchmark/results/association_vis/`](ai/benchmark/results/association_vis/)).*

Plain IoU is clearly too strict: plumes mostly sit beside or below the vehicle box, not inside it. The padded and centroid rules are tied on accuracy (0.938, i.e. one pair wrong each) — padded misses nothing but accepts one unrelated smoke box (1 false positive among 3 negatives); centroid rejects that one but misses one true pair. With only 16 pairs this cannot separate them, so the choice between them should be revisited on a larger labelled set; the user also reviews every report before it is shared.

## Reproducing the benchmark

See [`ai/benchmark/README.md`](ai/benchmark/README.md) for run order. `reproducibility.json` records the library versions, per-model training arguments, split hash and frozen thresholds used for the numbers above.

---

# Earlier training run (superseded) — visual artifacts

The images below come from the **earlier** Ultralytics training run on the
original, leaky split. They are retained for the qualitative illustration of
the annotation style and label distribution. **Do not quote precision, recall
or mAP from these plots as results**; use the benchmark section above.

## Training history

### Results across epochs

![YOLO training results](./ai/runs/detect/train/results.png)

### Dataset label distribution

![Dataset labels](./ai/runs/detect/train/labels.jpg)

## Detection performance

![Precision curve](./ai/runs/detect/train/BoxP_curve.png)

![Recall curve](./ai/runs/detect/train/BoxR_curve.png)

![F1 curve](./ai/runs/detect/train/BoxF1_curve.png)

![Precision-Recall curve](./ai/runs/detect/train/BoxPR_curve.png)

## Confusion matrices

![Confusion matrix](./ai/runs/detect/train/confusion_matrix.png)

![Normalized confusion matrix](./ai/runs/detect/train/confusion_matrix_normalized.png)

## Validation predictions

### Validation batch 0

![Validation batch 0 ground truth](./ai/runs/detect/train/val_batch0_labels.jpg)

![Validation batch 0 predictions](./ai/runs/detect/train/val_batch0_pred.jpg)

### Validation batch 1

![Validation batch 1 ground truth](./ai/runs/detect/train/val_batch1_labels.jpg)

![Validation batch 1 predictions](./ai/runs/detect/train/val_batch1_pred.jpg)

### Validation batch 2

![Validation batch 2 ground truth](./ai/runs/detect/train/val_batch2_labels.jpg)

![Validation batch 2 predictions](./ai/runs/detect/train/val_batch2_pred.jpg)

## Standalone validation pass

A separate validation run was also recorded under `ai/runs/detect/val/`:

![Validation precision curve](./ai/runs/detect/val/BoxP_curve.png)

![Validation recall curve](./ai/runs/detect/val/BoxR_curve.png)

![Validation F1 curve](./ai/runs/detect/val/BoxF1_curve.png)

![Validation PR curve](./ai/runs/detect/val/BoxPR_curve.png)

![Validation confusion matrix](./ai/runs/detect/val/confusion_matrix.png)

![Normalized validation confusion matrix](./ai/runs/detect/val/confusion_matrix_normalized.png)

![Validation batch 0 predictions](./ai/runs/detect/val/val_batch0_pred.jpg)

![Validation batch 1 predictions](./ai/runs/detect/val/val_batch1_pred.jpg)

![Validation batch 2 predictions](./ai/runs/detect/val/val_batch2_pred.jpg)

---

# Keeping the repository size manageable

The original dataset and trained model checkpoints were removed from the
repository because of their size.

The **result images and JSON/CSV files are small and are useful documentation**,
so they can be retained in Git without committing the dataset or `.pt`
checkpoints.

For GitHub to render the images above, the corresponding files must exist at
these repository-relative paths:

```text
System_architecture.png
comparison.png

ai/benchmark/results/
├── figures/
│   ├── fig2_benchmark.png
│   ├── fig3_pr_curves.png
│   ├── fig4_threshold_analysis.png
│   ├── fig5_accuracy_latency.png
│   └── fig6_qualitative.png
├── association_overview.jpg
├── association_vis/            # 12 overlay images
├── tables.md
├── test_metrics.json
├── thresholds.json
├── efficiency.json
├── efficiency_cpu.json
├── association_summary.json
├── association_candidates.csv
├── hard_cases_summary.json
├── hard_cases_detail.csv
├── split_audit.json
└── reproducibility.json

ai/runs/detect/train/           # earlier run, superseded
├── results.png
├── labels.jpg
├── BoxP_curve.png
├── BoxR_curve.png
├── BoxF1_curve.png
├── BoxPR_curve.png
├── confusion_matrix.png
├── confusion_matrix_normalized.png
├── val_batch0_labels.jpg
├── val_batch0_pred.jpg
├── val_batch1_labels.jpg
├── val_batch1_pred.jpg
├── val_batch2_labels.jpg
└── val_batch2_pred.jpg

ai/runs/detect/val/
├── BoxP_curve.png
├── BoxR_curve.png
├── BoxF1_curve.png
├── BoxPR_curve.png
├── confusion_matrix.png
├── confusion_matrix_normalized.png
├── val_batch0_pred.jpg
├── val_batch1_pred.jpg
└── val_batch2_pred.jpg
```

`ai/benchmark/results/preds/` (raw per-model prediction JSONs, ~1.3 MB) can also be kept for reproducibility of the metrics.

Your original local files for the earlier run are under:

```text
D:\PycharmProjects\AI_Smoke_Watch\ai\runs\detect\train\
D:\PycharmProjects\AI_Smoke_Watch\ai\runs\detect\val\
```

So **do not put those `D:\...` paths into the Markdown image links**. GitHub
cannot use your local Windows filesystem path. Copy the result files into the
repository while preserving the directory structure, then commit them.

# Screenshots

The repository include screenshots under:

```text
screenshots/
```

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

## Why the user reviews the report

The generated report is an assistive output.

The user sees the final content before it is handed to the X composer. The backend does not silently publish on the user's behalf.

---

# Known limitations

### 1. The shipped smoke model is not exhaust-specific

The current *shipped* (Tier 2) smoke detector comes from general fire/smoke imagery. It is therefore a **domain-transfer baseline**, not evidence of production-grade vehicle-exhaust detection.

The benchmark models above *were* trained on a smoke dataset, but their checkpoints are not committed to this repo, none has been wired into the backend yet, and the metrics apply to the benchmark test split only.

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

The difficult-case results above show this directly: fog and non-vehicle smoke triggered false positives in 2–3 of 4 negative images for four of the five models.

### 5. Benchmark limitations

- **Small test set and single seed.** 103 test images / 121 instances; one training run per model. Confidence intervals do not capture training variance, and they overlap across all models.
- **Near-duplicate images.** The grouped split keeps all augmented variants of one photo in the same split, which removes train/test leakage, but it means the test set itself contains near-duplicates (several images in Fig. 6 look like rotated or greyscale augmented variants). The effective sample size is therefore smaller than 103, and image-level bootstrap intervals are likely narrower than a group-level bootstrap would give.
- **Validation–test gap.** For every model, validation mAP@50 (0.37–0.42) is much lower than test mAP@50 (0.70–0.75). The split was audited and shows no leakage, so the likely explanation is that the test groups are easier or less varied than the validation groups, but this has not been verified. Test numbers should not be extrapolated to deployment.
- **Faster R-CNN threshold.** The frozen threshold (0.9) lies at the edge of the searched range and its scores are saturated (see Fig. 4 and Fig. 6), so its operating point is less well-characterised than the YOLO models'.
- **Latency is hardware-specific.** Reported on one Windows desktop (CPU with 4 threads, and a CUDA GPU). It is not on-device mobile latency.
- **Difficult cases and association sets are tiny** (10 images; 16 pairs) and hand-assembled. Treat them as indicative. The provenance of the images in `photos for test/` (real photographs vs. generated/composited images) should be stated wherever these results are cited.
- **Dataset provenance.** The smoke-only dataset merges three Roboflow sources; their licences and attribution requirements need to be verified before the data or models are redistributed.

### 6. Flutter still requires device/emulator validation

The mobile code needs to be built and exercised in a Flutter environment with the appropriate native permissions and device/emulator configuration.

---

# Research paper

A manuscript skeleton for the comparison lives in [`paper/`](paper/). Rules used to keep it honest: every number comes from `ai/benchmark/results/`; results are labelled as completed / validation-only / planned; and metrics from the earlier leaky-split run are not reported as benchmark results.

---

# Documentation

- [`docs/architecture.md`](docs/architecture.md) — system architecture and data flow
- [`docs/api.md`](docs/api.md) — API reference
- [`docs/demo.md`](docs/demo.md) — end-to-end demo procedure
- [`ai/benchmark/README.md`](ai/benchmark/README.md) — benchmark run order
- [`ai/benchmark/results/tables.md`](ai/benchmark/results/tables.md) — auto-generated benchmark tables
- `mobile/android/` — Android native project configuration in the supplied archive
- [`ai/weights/README.md`](ai/weights/README.md) — model provenance and weight tiers

---

# License

Add the project's intended license here before publishing the repository publicly.

If model weights or datasets are redistributed, also verify and document the licenses and attribution requirements of those individual assets.

---

## One-line summary

**SmokeWatch combines YOLO-based vehicle/smoke detection, spatial verification, license-plate OCR, location capture, and human-reviewed report generation into one end-to-end mobile workflow for documenting suspected visible vehicle exhaust smoke.**