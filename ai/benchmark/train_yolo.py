"""Train YOLOv8n/s and YOLO11n/s with ONE identical protocol on the grouped split.

    python train_yolo.py                       # all four, 50 epochs, batch 8, CPU
    python train_yolo.py --models yolov8n      # just one
    python train_yolo.py --device 0            # GPU (e.g. Colab); copy runs/ back afterwards
"""
import argparse
import os

from common import (MODELS, PRETRAINED, RUNS, IMGSZ, SEED, YOLO_KEYS, clear_pred_cache, ensure_dirs,
                    save_json, weights_path, write_data_yaml, env_info)


def is_complete(key, epochs):
    csv = RUNS / key / "results.csv"
    return weights_path(key).exists() and csv.exists() and len(csv.read_text().strip().splitlines()) - 1 >= epochs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=YOLO_KEYS, choices=YOLO_KEYS)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-amp", action="store_true", help="disable mixed precision (use if loss is NaN / mAP stays 0 on GTX 16xx)")
    ap.add_argument("--force", action="store_true", help="retrain even if a finished run exists")
    args = ap.parse_args()

    ensure_dirs()
    data = write_data_yaml()
    os.chdir(PRETRAINED)  # pretrained checkpoints (yolov8s.pt, yolo11n.pt, ...) auto-download here
    from ultralytics import YOLO

    for key in args.models:
        if is_complete(key, args.epochs) and not args.force:
            print(f"[skip] {key}: finished run found (use --force to retrain)")
            continue
        print(f"\n=== training {key} ===")
        cfg = dict(data=str(data), epochs=args.epochs, imgsz=IMGSZ, batch=args.batch, seed=SEED,
                   deterministic=True, optimizer="auto", lr0=0.01, weight_decay=0.0005, patience=100,
                   device=args.device, workers=args.workers, project=str(RUNS), name=key,
                   exist_ok=True, plots=True, amp=not args.no_amp)
        YOLO(MODELS[key]["ckpt"]).train(**cfg)
        clear_pred_cache(key)
        save_json({"model": key, "checkpoint": MODELS[key]["ckpt"], "train_args": cfg, "env": env_info()},
                  RUNS / key / "benchmark_train_config.json")


if __name__ == "__main__":
    main()
