### Table 1. Dataset (grouped, leakage-free split)

| Split | Images | Smoke instances |
|---|---:|---:|
| train | 728 | 816 |
| val | 195 | 260 |
| test | 103 | 121 |
| **Total** | **1026** | **1197** |

Original split leakage: 59 of 99 test images shared a source photo with train/val (77 source groups spanned several splits).

### Table 2. Held-out test results (thresholds frozen on validation; 95% bootstrap CI over test images)

| Model | Conf thr | Precision | Recall | F1 | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|---|---|
| YOLOv8n | 0.3 | 0.684 | 0.661 | 0.672 | 0.700 [0.601, 0.806] | 0.368 [0.308, 0.441] |
| YOLOv8s | 0.2 | 0.588 | 0.744 | 0.657 | 0.707 [0.610, 0.807] | 0.340 [0.281, 0.414] |
| YOLO11n | 0.3 | 0.740 | 0.636 | 0.684 | 0.720 [0.617, 0.824] | 0.364 [0.300, 0.435] |
| YOLO11s | 0.2 | 0.680 | 0.702 | 0.691 | 0.738 [0.636, 0.837] | 0.371 [0.311, 0.441] |
| Faster R-CNN (MobileNetV3-FPN) | 0.9 | 0.810 | 0.669 | 0.733 | 0.751 [0.666, 0.844] | 0.402 [0.342, 0.478] |

### Table 3. Efficiency (see efficiency.json for hardware/threads)

| Model | Params (M) | GFLOPs | File (MB) | Mean ms | Median ms | P95 ms | GPU mean ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8n | 3.01 | 8.2 | 6.2 | 51.7 | 50.8 | 57.1 | 20.5 |
| YOLOv8s | 11.14 | 28.6 | 22.5 | 127.5 | 122.3 | 157.5 | 27.3 |
| YOLO11n | 2.59 | 6.5 | 5.5 | 57.0 | 55.3 | 67.2 | 23.9 |
| YOLO11s | 9.43 | 21.7 | 19.2 | 121.5 | 118.9 | 136.9 | 26.1 |
| Faster R-CNN (MobileNetV3-FPN) | 18.95 | - | 76.0 | 128.7 | 124.1 | 159.7 | 29.9 |

Pareto-optimal (mAP@50-95 vs mean latency): YOLOv8n, YOLO11s, Faster R-CNN (MobileNetV3-FPN)

- Cross-check YOLOv8n: own evaluator mAP50=0.700/mAP50-95=0.368 vs Ultralytics 0.711/0.363
- Cross-check YOLOv8s: own evaluator mAP50=0.707/mAP50-95=0.340 vs Ultralytics 0.689/0.355
- Cross-check YOLO11n: own evaluator mAP50=0.720/mAP50-95=0.364 vs Ultralytics 0.725/0.366
- Cross-check YOLO11s: own evaluator mAP50=0.738/mAP50-95=0.371 vs Ultralytics 0.739/0.361

### Table 4. Hard cases (image level, frozen thresholds)

| Model | Smoke images | Recall | Negative images | False-positive rate |
|---|---:|---:|---:|---:|
| YOLOv8n | 6 | 0.500 | 4 | 0.750 |
| YOLOv8s | 6 | 0.833 | 4 | 0.750 |
| YOLO11n | 6 | 0.333 | 4 | 0.000 |
| YOLO11s | 6 | 0.833 | 4 | 0.500 |
| Faster R-CNN (MobileNetV3-FPN) | 6 | 0.500 | 4 | 0.500 |

Images flagged as smoke per category (detected / total):

| Category | YOLOv8n | YOLOv8s | YOLO11n | YOLO11s | Faster R-CNN (MobileNetV3-FPN) |
|---|---:|---:|---:|---:|---:|
| fog_mist | 1/1 | 1/1 | 0/1 | 0/1 | 1/1 |
| heavy_smoke | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| light_smoke | 0/2 | 1/2 | 0/2 | 1/2 | 1/2 |
| multi_vehicle | 1/2 | 2/2 | 0/2 | 2/2 | 0/2 |
| no_smoke | 0/1 | 0/1 | 0/1 | 0/1 | 0/1 |
| non_vehicle_smoke | 2/2 | 2/2 | 0/2 | 2/2 | 1/2 |
| smoke | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |

Box-level results per category (images with label files / verified negatives):

| Model | Category | Images | Precision | Recall | F1 | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|
| YOLOv8n | fog_mist | 1 | 0.000 | 0.000 | 0.000 | 1 | 0 |
| YOLOv8n | non_vehicle_smoke | 2 | 0.000 | 0.000 | 0.000 | 3 | 0 |
| YOLOv8n | no_smoke | 1 | 0.000 | 0.000 | 0.000 | 0 | 0 |
| YOLOv8s | fog_mist | 1 | 0.000 | 0.000 | 0.000 | 2 | 0 |
| YOLOv8s | non_vehicle_smoke | 2 | 0.000 | 0.000 | 0.000 | 3 | 0 |
| YOLOv8s | no_smoke | 1 | 0.000 | 0.000 | 0.000 | 0 | 0 |
| YOLO11n | fog_mist | 1 | 0.000 | 0.000 | 0.000 | 0 | 0 |
| YOLO11n | non_vehicle_smoke | 2 | 0.000 | 0.000 | 0.000 | 0 | 0 |
| YOLO11n | no_smoke | 1 | 0.000 | 0.000 | 0.000 | 0 | 0 |
| YOLO11s | fog_mist | 1 | 0.000 | 0.000 | 0.000 | 0 | 0 |
| YOLO11s | non_vehicle_smoke | 2 | 0.000 | 0.000 | 0.000 | 4 | 0 |
| YOLO11s | no_smoke | 1 | 0.000 | 0.000 | 0.000 | 0 | 0 |
| Faster R-CNN (MobileNetV3-FPN) | fog_mist | 1 | 0.000 | 0.000 | 0.000 | 1 | 0 |
| Faster R-CNN (MobileNetV3-FPN) | non_vehicle_smoke | 2 | 0.000 | 0.000 | 0.000 | 1 | 0 |
| Faster R-CNN (MobileNetV3-FPN) | no_smoke | 1 | 0.000 | 0.000 | 0.000 | 0 | 0 |

### Table 5. Vehicle-smoke association rules (16 labelled pairs, 13 positive)

| Rule | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|
| iou | 1.000 | 0.769 | 0.870 | 0.812 |
| centroid | 1.000 | 0.923 | 0.960 | 0.938 |
| padded | 0.929 | 1.000 | 0.963 | 0.938 |
| directional | 1.000 | 0.846 | 0.917 | 0.875 |