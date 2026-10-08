"""Uniform predictor interface + on-disk prediction cache for all benchmark models."""
import json

import cv2
import numpy as np

from common import IMGSZ, MODELS, PRED_DIR, ensure_dirs, list_images, norm_device, weights_path


class YoloPredictor:
    def __init__(self, weights, device="cpu"):
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        self.device = norm_device(device)

    def __call__(self, img_bgr, conf=0.001):
        r = self.model.predict(img_bgr, imgsz=IMGSZ, conf=conf, iou=0.7, max_det=300,
                               device=self.device, verbose=False)[0]
        return (r.boxes.xyxy.cpu().numpy().astype(np.float64),
                r.boxes.conf.cpu().numpy().astype(np.float64))

    def torch_model(self):
        return self.model.model


class FRCNNPredictor:
    def __init__(self, weights, device="cpu"):
        import torch
        from frcnn_model import build
        self.torch = torch
        ck = torch.load(str(weights), map_location="cpu", weights_only=True)
        self.model = build(ck["backbone"], pretrained=False)
        self.model.load_state_dict(ck["state"])
        self.device = norm_device(device)
        self.model.to(self.device).eval()

    def __call__(self, img_bgr, conf=0.001):
        torch = self.torch
        self.model.roi_heads.score_thresh = conf
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(self.device)
        with torch.no_grad():
            out = self.model([t])[0]
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize()
        return (out["boxes"].cpu().numpy().astype(np.float64),
                out["scores"].cpu().numpy().astype(np.float64))

    def torch_model(self):
        return self.model


def load_predictor(key, device="cpu"):
    w = weights_path(key)
    if not w.exists():
        raise FileNotFoundError(f"No trained weights for '{key}' at {w}")
    return YoloPredictor(w, device) if MODELS[key]["kind"] == "yolo" else FRCNNPredictor(w, device)


def get_predictions(key, split, device="cpu", force=False):
    """{image_name: {"boxes": [...], "scores": [...]}} at conf>=0.001, cached in results/preds/."""
    ensure_dirs()
    f = PRED_DIR / f"{key}_{split}.json"
    if f.exists() and not force:
        return json.loads(f.read_text())
    pred = load_predictor(key, device)
    out = {}
    for p in list_images(split):
        b, s = pred(cv2.imread(str(p)))
        out[p.name] = {"boxes": np.round(b, 2).tolist(), "scores": np.round(s, 5).tolist()}
    f.write_text(json.dumps(out))
    return out