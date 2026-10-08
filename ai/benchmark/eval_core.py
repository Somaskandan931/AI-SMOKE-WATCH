"""One shared evaluator for every model (YOLO and Faster R-CNN), so all numbers are computed identically.

AP follows the Ultralytics/COCO convention: 101-point interpolated precision envelope, IoU 0.50:0.05:0.95.
Precision/recall/F1 are computed at IoU 0.5 for a given confidence threshold.
"""
import cv2
import numpy as np

from common import SEED, list_images, load_gt_boxes

IOU_THRS = np.linspace(0.5, 0.95, 10)
_trapz = getattr(np, "trapezoid", None) or np.trapz


def box_iou(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def load_split_gt(split, root=None):
    """{image_name: (n,4) xyxy px}. Image size is read with cv2 (same loader the predictors use)."""
    gts = {}
    for p in list_images(split, root):
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        gts[p.name] = load_gt_boxes(p, w, h)
    return gts


def match_image(boxes, scores, gt):
    """Greedy matching by descending score. Returns (tp[n,10] bool, sorted_scores[n])."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    order = np.argsort(-scores, kind="stable")
    boxes, scores = boxes[order], scores[order]
    tp = np.zeros((len(boxes), len(IOU_THRS)), dtype=bool)
    if len(boxes) and len(gt):
        iou = box_iou(boxes, gt)
        for j, t in enumerate(IOU_THRS):
            used = np.zeros(len(gt), dtype=bool)
            for i in range(len(boxes)):
                cand = np.where((iou[i] >= t) & ~used)[0]
                if len(cand):
                    k = cand[np.argmax(iou[i, cand])]
                    used[k] = True
                    tp[i, j] = True
    return tp, scores


def build_stats(preds, gts):
    """preds: {name: {"boxes":[...], "scores":[...]}}; gts: {name: (n,4)} -> list of (tp, scores, n_gt)."""
    stats = []
    for name in sorted(gts):
        p = preds.get(name, {"boxes": [], "scores": []})
        tp, sc = match_image(p["boxes"], p["scores"], gts[name])
        stats.append((tp, sc, len(gts[name])))
    return stats


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(_trapz(np.interp(x, mrec, mpre), x))


def _pool(stats):
    tp = np.concatenate([s[0] for s in stats], axis=0)
    conf = np.concatenate([s[1] for s in stats])
    n_gt = int(sum(s[2] for s in stats))
    order = np.argsort(-conf, kind="stable")
    return tp[order], conf[order], n_gt


def _curve(tp_col, n_gt):
    tpc = np.cumsum(tp_col)
    fpc = np.cumsum(~tp_col)
    return tpc / max(n_gt, 1), tpc / np.maximum(tpc + fpc, 1e-9)


def summarize(stats, conf_thr):
    tp, conf, n_gt = _pool(stats)
    aps = []
    for j in range(len(IOU_THRS)):
        if n_gt == 0 or len(conf) == 0:
            aps.append(0.0)
            continue
        rec, prec = _curve(tp[:, j], n_gt)
        aps.append(compute_ap(rec, prec))
    keep = conf >= conf_thr
    TP = int(tp[keep, 0].sum())
    FP = int(keep.sum()) - TP
    FN = n_gt - TP
    p = TP / max(TP + FP, 1)
    r = TP / max(n_gt, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return {"ap50": aps[0], "ap50_95": float(np.mean(aps)), "precision": p, "recall": r, "f1": f1,
            "tp": TP, "fp": FP, "fn": FN, "n_gt": n_gt, "conf_thr": conf_thr}


def pr_curve(stats):
    """IoU=0.5 precision envelope sampled at 101 recall points (for Figure 3)."""
    tp, conf, n_gt = _pool(stats)
    rec, prec = _curve(tp[:, 0], n_gt)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([1.0], prec, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return {"recall": x.tolist(), "precision": np.interp(x, mrec, mpre).tolist()}


def bootstrap(stats, conf_thr, n_boot=1000, seed=SEED):
    """95% CI by resampling test IMAGES with replacement."""
    rng = np.random.default_rng(seed)
    n = len(stats)
    keys = ("ap50", "ap50_95", "precision", "recall", "f1")
    acc = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        s = summarize([stats[i] for i in idx], conf_thr)
        for k in keys:
            acc[k].append(s[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in acc.items()}
