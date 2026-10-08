"""Train every benchmark detector with the common protocol (see ai/configs/benchmark.yaml).

    python train_benchmark.py                    # YOLOv8n/s, YOLO11n/s, then Faster R-CNN (MobileNetV3-FPN)
    python train_benchmark.py --device 0 --frcnn-device cuda --frcnn-backbone resnet50
Run ai/benchmark/prepare_dataset.py first (leakage-free split).
"""
import argparse
import subprocess
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1] / "benchmark"

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu")
ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--skip-frcnn", action="store_true")
ap.add_argument("--frcnn-device", default=None)
ap.add_argument("--frcnn-backbone", default="mobilenet", choices=["mobilenet", "resnet50"])
ap.add_argument("--frcnn-epochs", type=int, default=30)
a = ap.parse_args()

subprocess.run([sys.executable, str(BENCH / "train_yolo.py"), "--device", a.device, "--epochs", str(a.epochs),
                "--batch", str(a.batch)], check=True)
if not a.skip_frcnn:
    subprocess.run([sys.executable, str(BENCH / "train_frcnn.py"), "--backbone", a.frcnn_backbone,
                    "--device", a.frcnn_device or ("cpu" if a.device == "cpu" else "cuda"),
                    "--epochs", str(a.frcnn_epochs)], check=True)
