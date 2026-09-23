"""
Run validation on a trained YOLO model and save the plots/metrics needed
for the README's "Model evaluation" section (PRD 14.10 / 17 Success
Metrics: precision, recall, mAP@50, mAP@50-95, per-class breakdown).

Ultralytics already writes confusion_matrix.png, PR_curve.png, F1_curve.png,
P_curve.png, R_curve.png, and results.png into its own run folder
(runs/detect/val<N>/) every time you call model.val() -- this script's job
is (1) to also copy the ones worth embedding into a stable, README-friendly
path (ai/training/results/), and (2) to add one chart Ultralytics doesn't
ship out of the box: a simple per-class precision/recall/mAP@50 bar chart,
plus a metrics.json / metrics.md you can paste straight into the README.

Usage:
    cd ai/training
    python visualize_eval.py --weights ../weights/best.pt --data ../configs/smoke.yaml

If you haven't trained a unified best.pt yet, point --weights at one of the
two community models instead to sanity-check the script, e.g.:
    python visualize_eval.py --weights ../weights/vehicle_yolov8n.pt --data ../configs/smoke.yaml
(Metrics from that run describe the source model on your data, not the
PRD-correct unified vehicle+exhaust_smoke model -- label the README image
accordingly if you do this.)
"""
import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless -- no display needed to save PNGs
import matplotlib.pyplot as plt

from ultralytics import YOLO

RESULTS_DIR = Path(__file__).parent / "results"


def copy_ultralytics_plots(val_save_dir: Path, dest: Path) -> list:
    """Ultralytics' own plots (only exist if model.val(plots=True), the
    default). Returns the list of filenames actually found and copied."""
    wanted = [
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "PR_curve.png",
        "F1_curve.png",
        "P_curve.png",
        "R_curve.png",
    ]
    copied = []
    for name in wanted:
        src = val_save_dir / name
        if src.exists():
            shutil.copy(src, dest / name)
            copied.append(name)
    return copied


def plot_per_class_bar(names: dict, precision, recall, map50, dest: Path):
    """A single bar chart: precision / recall / mAP@50 per class. Not
    something Ultralytics generates by default, and the one chart most
    useful for spotting a specific weak class (e.g. exhaust_smoke recall
    lagging vehicle recall) at a glance."""
    labels = [names[i] for i in sorted(names)]
    x = range(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([i - width for i in x], precision, width, label="Precision")
    ax.bar(list(x), recall, width, label="Recall")
    ax.bar([i + width for i in x], map50, width, label="mAP@50")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Per-class detection metrics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(dest / "per_class_metrics.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="../weights/best.pt")
    parser.add_argument("--data", default="../configs/smoke.yaml")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, split=args.split, plots=True)

    # --- overall metrics (what validate.py already prints) ---
    summary = {
        "weights": args.weights,
        "split": args.split,
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }

    # --- per-class breakdown ---
    names = metrics.names  # {class_id: class_name}
    per_class = {}
    try:
        p, r, ap50, ap = (
            metrics.box.p,
            metrics.box.r,
            metrics.box.ap50,
            metrics.box.ap,
        )
        for idx, cls_id in enumerate(metrics.box.ap_class_index):
            per_class[names[int(cls_id)]] = {
                "precision": float(p[idx]),
                "recall": float(r[idx]),
                "map50": float(ap50[idx]),
                "map50_95": float(ap[idx]),
            }
    except Exception as e:  # per-class arrays vary slightly by ultralytics version
        per_class = {"error": f"per-class breakdown unavailable: {e}"}

    summary["per_class"] = per_class

    # --- save machine-readable + README-ready outputs ---
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))

    md_lines = [
        f"# Evaluation results -- `{args.weights}` on `{args.split}` split\n",
        "| Metric | Value |",
        "|---|---|",
        f"| Precision | {summary['precision']:.3f} |",
        f"| Recall | {summary['recall']:.3f} |",
        f"| mAP@50 | {summary['map50']:.3f} |",
        f"| mAP@50-95 | {summary['map50_95']:.3f} |",
        "",
        "## Per-class",
        "",
        "| Class | Precision | Recall | mAP@50 | mAP@50-95 |",
        "|---|---|---|---|---|",
    ]
    for cls_name, m in per_class.items():
        if isinstance(m, dict):
            md_lines.append(
                f"| {cls_name} | {m['precision']:.3f} | {m['recall']:.3f} "
                f"| {m['map50']:.3f} | {m['map50_95']:.3f} |"
            )
    (RESULTS_DIR / "metrics.md").write_text("\n".join(md_lines))

    # --- plots ---
    val_save_dir = Path(metrics.save_dir)
    copied = copy_ultralytics_plots(val_save_dir, RESULTS_DIR)

    if per_class and "error" not in per_class:
        plot_per_class_bar(
            names,
            [per_class[names[int(i)]]["precision"] for i in metrics.box.ap_class_index],
            [per_class[names[int(i)]]["recall"] for i in metrics.box.ap_class_index],
            [per_class[names[int(i)]]["map50"] for i in metrics.box.ap_class_index],
            RESULTS_DIR,
        )
        copied.append("per_class_metrics.png")

    print(f"Precision: {summary['precision']:.3f}")
    print(f"Recall:    {summary['recall']:.3f}")
    print(f"mAP@50:    {summary['map50']:.3f}")
    print(f"mAP@50-95: {summary['map50_95']:.3f}")
    print(f"\nSaved to {RESULTS_DIR}/:")
    print(f"  metrics.json, metrics.md, {', '.join(copied) if copied else '(no plots -- ultralytics plots=True may have failed)'}")
    print(f"\nEmbed in README, e.g.: ![Confusion matrix]({RESULTS_DIR.relative_to(Path(__file__).parents[2])}/confusion_matrix.png)")


if __name__ == "__main__":
    main()
