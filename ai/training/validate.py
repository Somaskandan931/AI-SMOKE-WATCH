"""Run validation metrics (precision, recall, mAP@50, mAP@50-95) -- PRD 14.10."""
import argparse
from ultralytics import YOLO

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="../weights/best.pt")
    parser.add_argument("--data", default="../configs/smoke.yaml")
    args = parser.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(data=args.data)
    print("Precision:", metrics.box.mp)
    print("Recall:", metrics.box.mr)
    print("mAP@50:", metrics.box.map50)
    print("mAP@50-95:", metrics.box.map)
