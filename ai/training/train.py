"""
Train the vehicle + exhaust_smoke YOLO model (PRD section 14.5).

Usage:
    python train.py --data ../configs/smoke.yaml --model yolov8n.pt --epochs 100

Requires a Roboflow-exported dataset (see ../configs/smoke.yaml) with both
positive examples (vehicles with visible smoke, varied density/angle/light)
and negative/difficult examples (dust, fog, steam, shadows, blur) per PRD
section 14.3 -- negative examples matter most for avoiding false positives.
"""
import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../configs/smoke.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="base checkpoint to fine-tune from")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)

    # Ultralytics saves best.pt under runs/detect/train/weights/best.pt.
    # Where you copy it TO depends on which --data config you used:
    #   --data ../configs/smoke_only.yaml  (smoke-only fine-tune, the
    #       current recommended path -- see sources_smoke_only.yaml)
    #       -> copy to ai/weights/smoke_yolov8s.pt (OVERWRITES the
    #          existing fire/smoke model; config.py's two-model path
    #          picks it up automatically, vehicle model is untouched).
    #   --data ../configs/smoke.yaml  (2-class unified vehicle+smoke --
    #       only meaningful once a dataset with real vehicle-class boxes
    #       is in ai/dataset/processed/)
    #       -> copy to ai/weights/best.pt (find_trained_weights() then
    #          takes priority over BOTH separate models -- do not do
    #          this with a 2-class model trained on data that has zero
    #          vehicle boxes, it would silently disable vehicle
    #          detection entirely).


if __name__ == "__main__":
    main()