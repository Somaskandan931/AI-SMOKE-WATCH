"""Params / GFLOPs / file size / per-image latency (mean, median, P95) for every trained model.
Run this on the machine you want to quote in the paper (e.g. your laptop CPU).

    python measure_efficiency.py --threads 4
"""
import argparse
import time

import cv2
import numpy as np

from common import RESULTS, available_models, display_name, ensure_dirs, env_info, list_images, load_json, save_json, weights_path, IMGSZ
from predictors import load_predictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=None, help="torch CPU threads (record it in the paper)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()
    ensure_dirs()
    import torch
    if args.threads:
        torch.set_num_threads(args.threads)

    thr = load_json(RESULTS / "thresholds.json", default={})
    imgs = [cv2.imread(str(p)) for p in list_images("test")]
    res = {}
    for key in available_models():
        pred = load_predictor(key, args.device)
        tm = pred.torch_model()
        params = sum(p.numel() for p in tm.parameters())
        gflops = None
        if key != "frcnn":
            try:
                from ultralytics.utils.torch_utils import get_flops
                gflops = float(get_flops(tm, IMGSZ))
            except Exception:
                pass
        conf = thr.get(key, {}).get("threshold", 0.25)       # deployment-like NMS load, not conf=0.001
        for im in imgs[:args.warmup]:
            pred(im, conf)
        times = []
        for _ in range(args.repeats):
            for im in imgs:
                t0 = time.perf_counter()
                pred(im, conf)
                times.append((time.perf_counter() - t0) * 1000)
        t = np.array(times)
        res[key] = {"display": display_name(key), "params_m": params / 1e6, "gflops": gflops,
                    "file_mb": weights_path(key).stat().st_size / 1e6, "fp32_weights_mb": params * 4 / 1e6,
                    "latency_ms": {"mean": float(t.mean()), "median": float(np.median(t)),
                                   "p95": float(np.percentile(t, 95)), "n": int(len(t))},
                    "conf_used": conf}
        print(f"{res[key]['display']:28s} {params/1e6:6.2f}M params  mean {t.mean():7.1f} ms  p95 {np.percentile(t,95):7.1f} ms")
    save_json({"env": env_info() | {"device": args.device, "threads_set": args.threads,
                                    "protocol": "batch=1, imgsz 640, in-memory BGR image, includes pre/post-processing and NMS"},
               "models": res}, RESULTS / "efficiency.json")
    print(f"saved {RESULTS/'efficiency.json'}")


if __name__ == "__main__":
    main()
