"""Confidence-threshold experiment on the VALIDATION split only. Freezes one threshold per model
(max F1 @ IoU 0.5; ties -> higher threshold) into results/thresholds.json. Also draws Figure 4.

    python threshold_sweep.py
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import FIG_DIR, RESULTS, available_models, display_name, ensure_dirs, save_json
from eval_core import build_stats, load_split_gt, summarize
from predictors import get_predictions

THRS = [round(0.1 * i, 1) for i in range(1, 10)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--force", action="store_true", help="recompute cached predictions")
    args = ap.parse_args()
    ensure_dirs()

    keys = available_models()
    if not keys:
        raise SystemExit("No trained models found in runs/. Train first.")
    gts = load_split_gt("val")
    out, fig_data = {}, {}
    for key in keys:
        stats = build_stats(get_predictions(key, "val", args.device, args.force), gts)
        table = []
        for t in THRS:
            s = summarize(stats, t)
            table.append({k: s[k] for k in ("precision", "recall", "f1", "tp", "fp", "fn")} | {"thr": t})
        best = max(table, key=lambda r: (r["f1"], r["thr"]))
        full = summarize(stats, best["thr"])
        out[key] = {"display": display_name(key), "threshold": best["thr"], "val_f1_at_threshold": best["f1"],
                    "val_ap50": full["ap50"], "val_ap50_95": full["ap50_95"], "val_table": table}
        fig_data[key] = table
        print(f"{display_name(key):28s} frozen conf threshold = {best['thr']}  (val F1 {best['f1']:.3f})")
    save_json(out, RESULTS / "thresholds.json")

    n = len(keys)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.6 * rows), squeeze=False)
    for ax, key in zip(axes.flat, keys):
        t = fig_data[key]
        x = [r["thr"] for r in t]
        for m, lab in (("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")):
            ax.plot(x, [r[m] for r in t], marker="o", ms=3, label=lab)
        ax.axvline(out[key]["threshold"], ls="--", c="gray", lw=1)
        ax.set_title(display_name(key), fontsize=10)
        ax.set_xlabel("Confidence threshold"); ax.set_ylim(0, 1.02); ax.grid(alpha=.3)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_threshold_analysis.png", dpi=300)
    print(f"saved {RESULTS/'thresholds.json'} and fig4_threshold_analysis.png")


if __name__ == "__main__":
    main()
