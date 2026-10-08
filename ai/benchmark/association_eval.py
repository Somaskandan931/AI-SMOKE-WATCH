"""Compare vehicle-smoke association rules on candidate (vehicle, smoke) pairs that YOU label by eye.

    python association_eval.py dump  [--model auto]   # detect, write results/association_candidates.csv + annotated images
    # -> open results/association_vis/*.jpg, fill column gt_associated (1 = this smoke comes from this vehicle, 0 = not)
    python association_eval.py score                  # precision/recall/F1/accuracy per rule on your labelled pairs

Rules: iou (IoU>=0.05) | centroid (smoke centre within 0.25 vehicle-diagonals of the box) |
       padded (current backend rule: box padded 25% x / 60% y, any overlap) |
       directional (box padded 50% x, 60% up, 10% down; >=30% of the smoke box inside).
"""
import argparse
import csv
import math

import cv2

from common import AI_DIR, DATASET, REPO, RESULTS, SEED, ensure_dirs, load_json, save_json, list_images, IMG_EXT
from predictors import load_predictor

VEHICLES = {"car", "truck", "bus", "motorcycle"}
CSV_PATH = RESULTS / "association_candidates.csv"
RULES = ["iou", "centroid", "padded", "directional"]


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def r_iou(v, s):
    i = inter(v, s)
    return i / (area(v) + area(s) - i + 1e-9) >= 0.05


def r_centroid(v, s):
    cx, cy = (s[0] + s[2]) / 2, (s[1] + s[3]) / 2
    dx, dy = max(v[0] - cx, 0, cx - v[2]), max(v[1] - cy, 0, cy - v[3])
    return math.hypot(dx, dy) / (math.hypot(v[2] - v[0], v[3] - v[1]) + 1e-9) <= 0.25


def r_padded(v, s):  # identical to backend YoloService.boxes_associated
    w, h = v[2] - v[0], v[3] - v[1]
    e = (v[0] - 0.25 * w, v[1] - 0.6 * h, v[2] + 0.25 * w, v[3] + 0.6 * h)
    return min(e[2], s[2]) - max(e[0], s[0]) > 0 and min(e[3], s[3]) - max(e[1], s[1]) > 0


def r_directional(v, s):
    w, h = v[2] - v[0], v[3] - v[1]
    e = (v[0] - 0.5 * w, v[1] - 0.6 * h, v[2] + 0.5 * w, v[3] + 0.1 * h)
    return inter(e, s) / (area(s) + 1e-9) >= 0.30


FN = {"iou": r_iou, "centroid": r_centroid, "padded": r_padded, "directional": r_directional}


def pick_model(name):
    if name != "auto":
        return name
    tm = load_json(RESULTS / "test_metrics.json", default={})
    return max(tm, key=lambda k: tm[k]["test"]["ap50_95"]) if tm else "yolov8n"


def dump(a):
    ensure_dirs()
    key = pick_model(a.model)
    thr = load_json(RESULTS / "thresholds.json", default={}).get(key, {}).get("threshold", 0.25)
    smoke = load_predictor(key, a.device)
    from ultralytics import YOLO
    veh = YOLO(str(AI_DIR / "weights" / "vehicle_yolov8n.pt"))
    dirs = [DATASET / "images" / "test"] + [REPO / d for d in a.extra_dirs]
    vis = RESULTS / "association_vis"
    vis.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in dirs:
        for p in sorted(d.iterdir()) if d.exists() else []:
            if p.suffix.lower() not in IMG_EXT:
                continue
            img = cv2.imread(str(p))
            sb, ss = smoke(img, thr)
            r = veh.predict(img, imgsz=640, conf=a.vehicle_conf, device=a.device, verbose=False)[0]
            vb = [(b.xyxy[0].tolist(), float(b.conf[0]), r.names[int(b.cls[0])]) for b in r.boxes if r.names[int(b.cls[0])] in VEHICLES]
            if not len(ss) or not vb:
                continue
            canvas = img.copy()
            for i, (b, c, n) in enumerate(vb):
                cv2.rectangle(canvas, tuple(map(int, b[:2])), tuple(map(int, b[2:])), (0, 200, 0), 2)
                cv2.putText(canvas, f"V{i} {n} {c:.2f}", (int(b[0]), int(b[1]) + 18), 0, 0.6, (0, 200, 0), 2)
            for j, (b, c) in enumerate(zip(sb.tolist(), ss.tolist())):
                cv2.rectangle(canvas, tuple(map(int, b[:2])), tuple(map(int, b[2:])), (0, 0, 255), 2)
                cv2.putText(canvas, f"S{j} {c:.2f}", (int(b[0]), int(b[1]) + 18), 0, 0.6, (0, 0, 255), 2)
            cv2.imwrite(str(vis / f"{p.stem}.jpg"), canvas)
            for i, (vbx, vc, vn) in enumerate(vb):
                for j, (sbx, sc) in enumerate(zip(sb.tolist(), ss.tolist())):
                    rows.append([p.name, i, vn, round(vc, 3), j, round(float(sc), 3),
                                 *[int(FN[k](vbx, sbx)) for k in RULES], ""])
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "vehicle_idx", "vehicle_class", "vehicle_conf", "smoke_idx", "smoke_conf", *RULES, "gt_associated"])
        w.writerows(rows)
    print(f"smoke model: {key} @ conf {thr}; {len(rows)} candidate pairs -> {CSV_PATH}")
    print("Label gt_associated (1/0) by looking at results/association_vis/<image>.jpg (V# = vehicle, S# = smoke).")


def score():
    rows = [r for r in csv.DictReader(open(CSV_PATH)) if r["gt_associated"].strip() in ("0", "1")]
    if not rows:
        raise SystemExit("No labelled rows (fill gt_associated with 1/0)")
    out = {"n_labelled_pairs": len(rows), "n_positive": sum(r["gt_associated"].strip() == "1" for r in rows)}
    print(f"{len(rows)} labelled pairs ({out['n_positive']} positive)")
    for k in RULES:
        tp = fp = fn = tn = 0
        for r in rows:
            g, p = r["gt_associated"].strip() == "1", r[k] == "1"
            tp += g and p; fp += (not g) and p; fn += g and (not p); tn += (not g) and (not p)
        P, R = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        out[k] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": P, "recall": R,
                  "f1": 2 * P * R / max(P + R, 1e-9), "accuracy": (tp + tn) / len(rows)}
        print(f"{k:12s} P={P:.3f} R={R:.3f} F1={out[k]['f1']:.3f} acc={out[k]['accuracy']:.3f}")
    save_json(out, RESULTS / "association_summary.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dump", "score"])
    ap.add_argument("--model", default="auto")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--vehicle-conf", type=float, default=0.5)
    ap.add_argument("--extra-dirs", nargs="*", default=["photos for test"], help="folders relative to repo root")
    a = ap.parse_args()
    dump(a) if a.cmd == "dump" else score()
