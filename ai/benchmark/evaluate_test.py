"""Final held-out TEST evaluation with the frozen thresholds from threshold_sweep.py.

    python evaluate_test.py --crosscheck      # also re-computes mAP with Ultralytics' own validator (YOLO models)
"""
import argparse

from common import RESULTS, available_models, display_name, ensure_dirs, load_json, save_json, weights_path, write_data_yaml
from eval_core import bootstrap, build_stats, load_split_gt, pr_curve, summarize
from predictors import get_predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--crosscheck", action="store_true")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    ensure_dirs()

    thr = load_json(RESULTS / "thresholds.json")
    gts_test, gts_val = load_split_gt("test"), load_split_gt("val")
    out = {}
    for key in available_models():
        if key not in thr:
            print(f"[skip] {key}: no frozen threshold (rerun threshold_sweep.py)")
            continue
        t = thr[key]["threshold"]
        st = build_stats(get_predictions(key, "test", args.device, args.force), gts_test)
        sv = build_stats(get_predictions(key, "val", args.device, args.force), gts_val)
        rec = {"display": display_name(key), "threshold": t, "test": summarize(st, t),
               "val": summarize(sv, t), "test_ci95": bootstrap(st, t, args.boot), "pr_test": pr_curve(st)}
        if args.crosscheck and key != "frcnn":
            try:
                from ultralytics import YOLO
                m = YOLO(str(weights_path(key))).val(data=str(write_data_yaml()), split="test", imgsz=640,
                                                     conf=0.001, iou=0.7, device=args.device, plots=False, verbose=False)
                rec["ultralytics_crosscheck"] = {"map50": float(m.box.map50), "map50_95": float(m.box.map)}
            except Exception as e:
                rec["ultralytics_crosscheck"] = {"error": str(e)}
        out[key] = rec
        s = rec["test"]
        print(f"{rec['display']:28s} thr={t}  P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1']:.3f} "
              f"mAP50={s['ap50']:.3f} mAP50-95={s['ap50_95']:.3f}"
              + (f"  | ultralytics: {rec['ultralytics_crosscheck']}" if 'ultralytics_crosscheck' in rec else ""))
    save_json(out, RESULTS / "test_metrics.json")
    print(f"saved {RESULTS/'test_metrics.json'}")


if __name__ == "__main__":
    main()
