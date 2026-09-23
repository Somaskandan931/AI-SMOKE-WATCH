# Model weights

`yolo_service.py` and `plate_detector.py` check for weight files here on
startup, in this priority order, and load whichever tier is present —
no code or config changes needed either way:

1. **`best.pt`** or **`last.pt`** — a unified vehicle + exhaust_smoke
   detector, produced by training `ai/training/train.py` on the merged
   dataset (`ai/configs/smoke.yaml`). Not present until you train one —
   see `ai/dataset/sources.yaml` and `ai/dataset/download_from_roboflow.py`.
   Once this exists it takes priority over everything below.

2. **`vehicle_yolov8n.pt`** + **`smoke_yolov8s.pt`** — ship with this repo
   already. Real, separately-sourced pretrained models standing in for
   the unified model above until you've trained one:
   - `vehicle_yolov8n.pt` — stock Ultralytics YOLOv8n (COCO); its
     car/truck/bus/motorcycle classes are relabeled to `"vehicle"`.
   - `smoke_yolov8s.pt` — a community model trained on general fire/smoke
     imagery, not exhaust-specific; its `"smoke"` class is relabeled to
     `"exhaust_smoke"`. Expect lower precision on real exhaust plumes
     than a model actually trained on vehicle exhaust would give.

3. **Mock** — an OpenCV heuristic, used only if neither tier above loads
   (e.g. `ultralytics` isn't installed, or no weight files are present
   at all).

`plate_best.pt` — a separate license-plate detector, ships with this
repo, used by `/api/plate/detect`. Falls back to an OpenCV contour
heuristic if missing.

Check which tier is actually active — and the exact class names each
loaded model reports — by hitting `GET /api/health`:
```json
{
  "yolo_mode": "two_model",
  "yolo_diagnostics": {"vehicle_classes": {...}, "smoke_classes": {...}},
  "plate_mode": "model"
}
```
If `yolo_diagnostics` shows classes that don't include `"car"`/`"smoke"`
(or whatever `config.py`'s `VEHICLE_CLASS_NAMES`/`SMOKE_CLASS_NAME`
expect), the server log will also print an explicit warning at startup —
that mismatch means detection for that class silently never fires, so
don't assume "no smoke detected" means the photo had none until you've
confirmed `yolo_mode` isn't quietly `"mock"` and the class names line up.
