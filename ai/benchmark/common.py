"""Shared paths, constants and helpers for the smoke-detector benchmark."""
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

BENCH = Path(__file__).resolve().parent
AI_DIR = BENCH.parent
REPO = AI_DIR.parent
SRC_DATASET = AI_DIR / "dataset" / "processed_smoke_only"          # original split (has leakage, see prepare_dataset.py)
DATASET = AI_DIR / "dataset" / "processed_smoke_only_grouped"      # leakage-free split used for ALL models
RUNS = BENCH / "runs"
RESULTS = BENCH / "results"
PRED_DIR = RESULTS / "preds"
FIG_DIR = RESULTS / "figures"
PRETRAINED = BENCH / "pretrained"

SEED = 0
IMGSZ = 640
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MODELS = {
    "yolov8n": {"display": "YOLOv8n", "kind": "yolo", "ckpt": "yolov8n.pt"},
    "yolov8s": {"display": "YOLOv8s", "kind": "yolo", "ckpt": "yolov8s.pt"},
    "yolo11n": {"display": "YOLO11n", "kind": "yolo", "ckpt": "yolo11n.pt"},
    "yolo11s": {"display": "YOLO11s", "kind": "yolo", "ckpt": "yolo11s.pt"},
    "frcnn": {"display": "Faster R-CNN", "kind": "frcnn", "ckpt": None},
}
YOLO_KEYS = [k for k, v in MODELS.items() if v["kind"] == "yolo"]


def norm_device(d):
    """'0' -> 'cuda:0', 'cuda' -> 'cuda:0', 'cpu' stays 'cpu'."""
    d = str(d)
    if d.isdigit():
        return f"cuda:{d}"
    return "cuda:0" if d == "cuda" else d


def ensure_dirs():
    for d in (RUNS, RESULTS, PRED_DIR, FIG_DIR, PRETRAINED):
        d.mkdir(parents=True, exist_ok=True)


def weights_path(key):
    if MODELS[key]["kind"] == "yolo":
        return RUNS / key / "weights" / "best.pt"
    return RUNS / key / "best.pt"


def available_models():
    return [k for k in MODELS if weights_path(k).exists()]


def display_name(key):
    if key == "frcnn":
        cfg = RUNS / "frcnn" / "train_args.json"
        if cfg.exists():
            bb = json.loads(cfg.read_text()).get("backbone", "")
            nice = {"mobilenet": "MobileNetV3-FPN", "resnet50": "ResNet50-FPN"}.get(bb, bb)
            return f"Faster R-CNN ({nice})"
    return MODELS[key]["display"]


def clear_pred_cache(key):
    for f in PRED_DIR.glob(f"{key}_*.json"):
        f.unlink()


def write_data_yaml():
    """Absolute-path dataset yaml, so Ultralytics never has to guess what a relative path means."""
    p = BENCH / "data.yaml"
    p.write_text(
        f'path: "{DATASET.as_posix()}"\ntrain: images/train\nval: images/val\ntest: images/test\n'
        "names:\n  0: smoke\n"
    )
    return p


def list_images(split, root=None):
    d = (root or DATASET) / "images" / split
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)


def label_path(img_path):
    # <root>/images/<split>/x.jpg -> <root>/labels/<split>/x.txt
    return img_path.parents[2] / "labels" / img_path.parent.name / (img_path.stem + ".txt")


def read_label_file(lp, w, h):
    """YOLO txt (boxes or polygons) -> (n,4) float32 xyxy in pixels. Single class, so class id is ignored."""
    boxes = []
    lp = Path(lp)
    if lp.exists():
        for line in lp.read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            if len(p) == 5:
                cx, cy, bw, bh = map(float, p[1:5])
            else:
                xs = list(map(float, p[1::2]))
                ys = list(map(float, p[2::2]))
                x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
                cx, cy, bw, bh = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
            boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(boxes, dtype=np.float32).reshape(-1, 4)


def load_gt_boxes(img_path, w, h):
    return read_label_file(label_path(img_path), w, h)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2))


_MISSING = object()


def load_json(path, default=_MISSING):
    """Load JSON. If the file is missing: return `default` when one was passed (even None), else raise."""
    p = Path(path)
    if not p.exists():
        if default is not _MISSING:
            return default
        raise FileNotFoundError(f"{p} not found - run the previous pipeline step first (see README.md)")
    return json.loads(p.read_text())


def env_info():
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["torch_threads"] = torch.get_num_threads()
    except Exception:
        pass
    try:
        import torchvision
        info["torchvision"] = torchvision.__version__
    except Exception:
        pass
    try:
        import ultralytics
        info["ultralytics"] = ultralytics.__version__
    except Exception:
        pass
    return info