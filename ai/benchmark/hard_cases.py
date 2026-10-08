"""Difficult-case / hard-negative check on hand-labelled images (fog, dust, steam, no-smoke, multi-vehicle ...).

    python hard_cases.py init     # builds hard_cases/manifest.csv from photos for test/, hard_cases/images/, raw/negatives/
    # -> open manifest.csv, LOOK at every image, fix has_smoke (1/0) + category, set verified=1
    python hard_cases.py run      # image-level results at each model's frozen threshold

The filename-based guesses written by `init` are only a starting point and are NOT used until verified=1.
Optional box-level metrics: put a YOLO label file next to an image (same name, .txt) -> per-category precision/recall/F1.
Verified has_smoke=0 images count as "ground truth = no boxes", so every box on them is a false positive.
Image-level rule: an image is 'detected' if any box has confidence >= the model's frozen threshold.
"""
import argparse
import csv
from collections import defaultdict

import cv2
import numpy as np

from common import (AI_DIR, BENCH, IMG_EXT, REPO, RESULTS, available_models, display_name, ensure_dirs,
                    load_json, read_label_file, save_json)
from eval_core import match_image, summarize
from predictors import load_predictor

MANIFEST = BENCH / "hard_cases" / "manifest.csv"
SRC_DIRS = [REPO / "photos for test", BENCH / "hard_cases" / "images", AI_DIR / "dataset" / "hard_negatives",
            AI_DIR / "dataset" / "raw" / "negatives"]


def guess(name):
    n = name.lower()
    for kw, cat, s in (("no smoke", "no_smoke", "0"), ("mist", "fog_mist", "0"), ("fog", "fog_mist", "0"),
                       ("dust", "dust", "0"), ("steam", "steam", "0"), ("bin", "non_vehicle_smoke", "1"),
                       ("heavy", "heavy_smoke", "1"), ("light", "light_smoke", "1"),
                       ("smok", "smoke", "1"), ("exhaust", "smoke", "1"),
                       ("traffic", "multi_vehicle", ""), ("highway", "multi_vehicle", ""), ("truck", "multi_vehicle", "")):
        if kw in n:
            return cat, s
    return "other", ""


def init(force):
    if MANIFEST.exists() and not force:
        raise SystemExit(f"{MANIFEST} exists (use --force to overwrite - you will lose your edits)")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    (BENCH / "hard_cases" / "images").mkdir(exist_ok=True)
    rows = []
    for d in SRC_DIRS:
        if d.exists():
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in IMG_EXT:
                    cat, s = guess(p.name)
                    rows.append([p.relative_to(REPO).as_posix(), cat, s, 0])
    with open(MANIFEST, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "category", "has_smoke", "verified"])
        w.writerows(rows)
    print(f"wrote {MANIFEST} with {len(rows)} rows - edit it, then set verified=1 per row")


def _brief(s):
    return {k: s[k] for k in ("precision", "recall", "f1", "tp", "fp", "fn", "n_gt")}


def run(device):
    ensure_dirs()
    thr = load_json(RESULTS / "thresholds.json")
    rows = [r for r in csv.DictReader(open(MANIFEST)) if r["verified"].strip() == "1" and r["has_smoke"].strip() in ("0", "1")]
    if not rows:
        raise SystemExit("No verified rows in manifest.csv")
    imgs, gts = {}, {}
    for r in rows:
        p = REPO / r["path"]
        im = cv2.imread(str(p))
        imgs[r["path"]] = im
        h, w = im.shape[:2]
        if p.with_suffix(".txt").exists():                     # optional YOLO sidecar label -> box-level metrics
            gts[r["path"]] = read_label_file(p.with_suffix(".txt"), w, h)
        elif r["has_smoke"].strip() == "0":                    # verified negative: ground truth = no boxes
            gts[r["path"]] = np.zeros((0, 4), np.float32)
    summary, detail = {}, []
    for key in available_models():
        if key not in thr:
            continue
        t = thr[key]["threshold"]
        pred = load_predictor(key, device)
        cm = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        per_cat = defaultdict(lambda: {"n": 0, "detected": 0})
        box_cat, box_all = defaultdict(list), []
        for r in rows:
            b, sc = pred(imgs[r["path"]], 0.001)
            mx = float(sc.max()) if len(sc) else 0.0
            det, truth = mx >= t, r["has_smoke"].strip() == "1"
            cm[("tp" if truth else "fp") if det else ("fn" if truth else "tn")] += 1
            per_cat[r["category"]]["n"] += 1
            per_cat[r["category"]]["detected"] += int(det)
            if r["path"] in gts:
                tp, s_sorted = match_image(b, sc, gts[r["path"]])
                st = (tp, s_sorted, len(gts[r["path"]]))
                box_cat[r["category"]].append(st)
                box_all.append(st)
            detail.append({"model": key, "path": r["path"], "category": r["category"], "has_smoke": int(truth),
                           "max_score": round(mx, 4), "detected": int(det)})
        neg, pos = cm["fp"] + cm["tn"], cm["tp"] + cm["fn"]
        summary[key] = {"display": display_name(key), "threshold": t, **cm,
                        "false_positive_rate": cm["fp"] / neg if neg else None,
                        "recall_on_smoke_images": cm["tp"] / pos if pos else None,
                        "n_smoke_images": pos, "n_negative_images": neg, "per_category": dict(per_cat),
                        "box_level": {"overall": _brief(summarize(box_all, t)) if box_all else None,
                                      "per_category": {c: {**_brief(summarize(v, t)), "n_images": len(v)}
                                                       for c, v in box_cat.items()}}}
        print(f"{display_name(key):28s} FP-rate {summary[key]['false_positive_rate']}  recall {summary[key]['recall_on_smoke_images']}  {cm}")
    save_json(summary, RESULTS / "hard_cases_summary.json")
    with open(RESULTS / "hard_cases_detail.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(detail[0].keys()))
        w.writeheader(); w.writerows(detail)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["init", "run"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    init(a.force) if a.cmd == "init" else run(a.device)
