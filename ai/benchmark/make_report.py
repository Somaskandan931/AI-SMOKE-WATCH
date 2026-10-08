"""Turn results/*.json into paper-ready tables (results/tables.md) and Figures 2, 3, 5, 6.
Figure 4 comes from threshold_sweep.py.   python make_report.py [--model auto]
"""
import argparse

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import FIG_DIR, RESULTS, display_name, ensure_dirs, list_images, load_json, load_gt_boxes
from eval_core import box_iou
from predictors import get_predictions


def fmt_ci(v, ci):
    return f"{v:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def pareto(points):
    """points: {key: (quality, cost)}; non-dominated = no other model has >= quality AND <= cost (one strict)."""
    front = []
    for k, (q, c) in points.items():
        if not any((q2 >= q and c2 <= c) and (q2 > q or c2 < c) for k2, (q2, c2) in points.items() if k2 != k):
            front.append(k)
    return front


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="auto", help="model used for the qualitative figure")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    ensure_dirs()
    tm = load_json(RESULTS / "test_metrics.json")
    cpu_f = RESULTS / "efficiency_cpu.json"          # CPU run = mobile-relevant latency; efficiency.json = GPU run
    eff = load_json(cpu_f if cpu_f.exists() else RESULTS / "efficiency.json", default={"models": {}})["models"]
    gpu = load_json(RESULTS / "efficiency.json", default={"models": {}})["models"] if cpu_f.exists() else {}
    audit = load_json(RESULTS / "split_audit.json", default={})
    keys = list(tm)
    L = []

    if "counts" in audit:
        L += ["### Table 1. Dataset (grouped, leakage-free split)", "", "| Split | Images | Smoke instances |", "|---|---:|---:|"]
        tot_i = tot_b = 0
        for s in ("train", "val", "test"):
            c = audit["counts"][s]; tot_i += c["images"]; tot_b += c["smoke_instances"]
            L.append(f"| {s} | {c['images']} | {c['smoke_instances']} |")
        L += [f"| **Total** | **{tot_i}** | **{tot_b}** |", ""]
        b = audit["before"]
        L += [f"Original split leakage: {b['test']['images_sharing_a_source_with_another_split']} of {b['test']['images']} test images "
              f"shared a source photo with train/val ({b['groups_spanning_multiple_splits']} source groups spanned several splits).", ""]

    L += ["### Table 2. Held-out test results (thresholds frozen on validation; 95% bootstrap CI over test images)", "",
          "| Model | Conf thr | Precision | Recall | F1 | mAP@50 | mAP@50-95 |", "|---|---:|---:|---:|---:|---|---|"]
    for k in keys:
        r, ci, s = tm[k], tm[k]["test_ci95"], tm[k]["test"]
        L.append(f"| {r['display']} | {r['threshold']} | {s['precision']:.3f} | {s['recall']:.3f} | {s['f1']:.3f} | "
                 f"{fmt_ci(s['ap50'], ci['ap50'])} | {fmt_ci(s['ap50_95'], ci['ap50_95'])} |")
    L.append("")
    if eff:
        L += ["### Table 3. Efficiency (see efficiency.json for hardware/threads)", "",
              "| Model | Params (M) | GFLOPs | File (MB) | Mean ms | Median ms | P95 ms |" + (" GPU mean ms |" if gpu else ""),
              "|---|---:|---:|---:|---:|---:|---:|" + ("---:|" if gpu else "")]
        for k in keys:
            if k in eff:
                e = eff[k]; l = e["latency_ms"]
                g = f"{e['gflops']:.1f}" if e["gflops"] else "-"
                L.append(f"| {e['display']} | {e['params_m']:.2f} | {g} | {e['file_mb']:.1f} | {l['mean']:.1f} | {l['median']:.1f} | {l['p95']:.1f} |"
                         + (f" {gpu[k]['latency_ms']['mean']:.1f} |" if k in gpu else (" - |" if gpu else "")))
        pts = {k: (tm[k]["test"]["ap50_95"], eff[k]["latency_ms"]["mean"]) for k in keys if k in eff}
        L += ["", "Pareto-optimal (mAP@50-95 vs mean latency): " + ", ".join(display_name(k) for k in pareto(pts)), ""]
    for k in keys:
        cc = tm[k].get("ultralytics_crosscheck")
        if cc and "map50" in cc:
            L.append(f"- Cross-check {tm[k]['display']}: own evaluator mAP50={tm[k]['test']['ap50']:.3f}/mAP50-95={tm[k]['test']['ap50_95']:.3f} vs Ultralytics {cc['map50']:.3f}/{cc['map50_95']:.3f}")
    hc = load_json(RESULTS / "hard_cases_summary.json", default=None)
    if not hc:
        print("note: hard_cases_summary.json not found - skipping Table 4 (run hard_cases.py after verifying manifest.csv)")
    if hc:
        L += ["", "### Table 4. Hard cases (image level, frozen thresholds)", "",
              "| Model | Smoke images | Recall | Negative images | False-positive rate |", "|---|---:|---:|---:|---:|"]
        for k, v in hc.items():
            rr, fp = v["recall_on_smoke_images"], v["false_positive_rate"]
            L.append(f"| {v['display']} | {v['n_smoke_images']} | {'-' if rr is None else f'{rr:.3f}'} | "
                     f"{v['n_negative_images']} | {'-' if fp is None else f'{fp:.3f}'} |")
        cats = sorted({c for v in hc.values() for c in v["per_category"]})
        L += ["", "Images flagged as smoke per category (detected / total):", "",
              "| Category | " + " | ".join(v["display"] for v in hc.values()) + " |", "|---|" + "---:|" * len(hc)]
        for c in cats:
            L.append(f"| {c} | " + " | ".join(
                f"{v['per_category'][c]['detected']}/{v['per_category'][c]['n']}" if c in v["per_category"] else "-"
                for v in hc.values()) + " |")
        if any(v.get("box_level", {}).get("per_category") for v in hc.values()):
            L += ["", "Box-level results per category (images with label files / verified negatives):", "",
                  "| Model | Category | Images | Precision | Recall | F1 | FP | FN |", "|---|---|---:|---:|---:|---:|---:|---:|"]
            for k, v in hc.items():
                for c, m in v["box_level"]["per_category"].items():
                    L.append(f"| {v['display']} | {c} | {m['n_images']} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['fp']} | {m['fn']} |")
    asc = load_json(RESULTS / "association_summary.json", default=None)
    if not asc:
        print("note: association_summary.json not found - skipping Table 5 (label gt_associated, then run association_eval.py score)")
    if asc:
        L += ["", f"### Table 5. Vehicle-smoke association rules ({asc['n_labelled_pairs']} labelled pairs, {asc['n_positive']} positive)", "",
              "| Rule | Precision | Recall | F1 | Accuracy |", "|---|---:|---:|---:|---:|"]
        for k, m in asc.items():
            if isinstance(m, dict):
                L.append(f"| {k} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['accuracy']:.3f} |")
    (RESULTS / "tables.md").write_text("\n".join(L))
    print("wrote tables.md")

    names = [tm[k]["display"] for k in keys]
    x = np.arange(len(keys)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(x - w / 2, [tm[k]["test"]["ap50"] for k in keys], w, label="mAP@50")
    ax.bar(x + w / 2, [tm[k]["test"]["ap50_95"] for k in keys], w, label="mAP@50-95")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15, fontsize=8); ax.set_ylabel("Score"); ax.set_ylim(0, 1); ax.legend(); ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig2_benchmark.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4))
    for k in keys:
        ax.plot(tm[k]["pr_test"]["recall"], tm[k]["pr_test"]["precision"], label=tm[k]["display"])
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision (IoU 0.5)"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig3_pr_curves.png", dpi=300); plt.close(fig)

    if eff:
        fig, ax = plt.subplots(figsize=(5.2, 4))
        for k in keys:
            if k in eff:
                ax.scatter(eff[k]["latency_ms"]["mean"], tm[k]["test"]["ap50_95"], s=40)
                ax.annotate(tm[k]["display"], (eff[k]["latency_ms"]["mean"], tm[k]["test"]["ap50_95"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("Mean latency per image (ms, CPU)" if cpu_f.exists() else "Mean latency per image (ms)"); ax.set_ylabel("Test mAP@50-95"); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(FIG_DIR / "fig5_accuracy_latency.png", dpi=300); plt.close(fig)

    # ---- Figure 6: qualitative examples from the selected model on the test split ----
    key = max(keys, key=lambda k: tm[k]["test"]["ap50_95"]) if a.model == "auto" else a.model
    thr = tm[key]["threshold"]
    preds = get_predictions(key, "test", a.device)
    cats = {"Correct (best IoU)": None, "Missed smoke": None, "False / extra box": None, "Smallest smoke": None}
    best = {k: -1 for k in cats}
    for p in list_images("test"):
        img = cv2.imread(str(p)); h, wd = img.shape[:2]
        gt = load_gt_boxes(p, wd, h)
        pr = preds[p.name]
        pb = np.array(pr["boxes"]).reshape(-1, 4); ps = np.array(pr["scores"])
        keep = ps >= thr; pb, ps = pb[keep], ps[keep]
        iou = box_iou(pb, gt)
        mean_iou = float(iou.max(axis=0).mean()) if iou.size else 0.0
        if len(gt) and len(pb) and mean_iou > best["Correct (best IoU)"]:
            best["Correct (best IoU)"], cats["Correct (best IoU)"] = mean_iou, (p, gt, pb, ps)
        if len(gt) and (iou.size == 0 or iou.max() < 0.5):
            ga = float(max((g[2] - g[0]) * (g[3] - g[1]) for g in gt))
            if ga > best["Missed smoke"]:
                best["Missed smoke"], cats["Missed smoke"] = ga, (p, gt, pb, ps)
        if len(pb) and (iou.size == 0 or (iou.max(axis=1) < 0.5).any()):
            if float(ps.max()) > best["False / extra box"]:
                best["False / extra box"], cats["False / extra box"] = float(ps.max()), (p, gt, pb, ps)
        if len(gt):
            fr = -float(min((g[2] - g[0]) * (g[3] - g[1]) for g in gt)) / (wd * h)
            if best["Smallest smoke"] == -1 or fr > best["Smallest smoke"]:
                best["Smallest smoke"], cats["Smallest smoke"] = fr, (p, gt, pb, ps)
    fig, axes = plt.subplots(2, 2, figsize=(8, 6.4))
    for ax, (title, item) in zip(axes.flat, cats.items()):
        ax.axis("off"); ax.set_title(title, fontsize=9)
        if item is None:
            ax.text(0.5, 0.5, "no example", ha="center"); continue
        p, gt, pb, ps = item
        ax.imshow(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB))
        for g in gt:
            ax.add_patch(plt.Rectangle((g[0], g[1]), g[2] - g[0], g[3] - g[1], fill=False, ec="lime", lw=2))
        for b, s in zip(pb, ps):
            ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="red", lw=2))
            ax.text(b[0], b[1], f"{s:.2f}", color="red", fontsize=8, va="bottom")
    fig.suptitle(f"{display_name(key)} - green: ground truth, red: prediction (conf >= {thr})", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig6_qualitative.png", dpi=300)
    print(f"figures saved to {FIG_DIR}")

    # ---- reproducibility record + copy artefacts into paper/ ----
    import hashlib, shutil
    from common import AI_DIR, REPO, RUNS, DATASET
    manifest = DATASET.parent / "grouped_split_manifest.csv"
    repro = {"env": eff and load_json(RESULTS / "efficiency.json")["env"],
             "split_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.exists() else None,
             "split_counts": audit.get("counts"), "seed": 0, "imgsz": 640,
             "train_configs": {d.name: load_json(d / "benchmark_train_config.json", default={}) for d in RUNS.glob("yolo*")}
             | {"frcnn": load_json(RUNS / "frcnn" / "train_args.json", default={})},
             "frozen_thresholds": {k: v["threshold"] for k, v in load_json(RESULTS / "thresholds.json", default={}).items()}}
    (RESULTS / "reproducibility.json").write_text(__import__("json").dumps(repro, indent=2))
    pd = REPO / "paper"
    (pd / "figures").mkdir(parents=True, exist_ok=True); (pd / "tables").mkdir(exist_ok=True)
    for f in FIG_DIR.glob("fig*.png"):
        shutil.copy2(f, pd / "figures" / f.name)
    shutil.copy2(RESULTS / "tables.md", pd / "tables" / "tables.md")
    shutil.copy2(RESULTS / "reproducibility.json", pd / "tables" / "reproducibility.json")
    print(f"copied figures/tables to {pd}")


if __name__ == "__main__":
    main()