# SmokeWatch detector benchmark (paper experiments)

Run everything from this folder (`ai/benchmark`) with the Anaconda Python that already has `ultralytics`.
Nothing here modifies the app, the old training scripts or the original dataset.

```powershell
cd D:\PycharmProjects\AI_Smoke_Watch\ai\benchmark

# 0. Leakage audit, then build the clean split (ai/dataset/processed_smoke_only_grouped)
python prepare_dataset.py --audit-only
python prepare_dataset.py

# 1. Train (identical protocol: 640px, 50 ep, batch 8, seed 0, optimizer auto, lr0 0.01, wd 5e-4)
python train_yolo.py                          # YOLOv8n/s, YOLO11n/s   (CPU: roughly 3-5 h total)
python train_frcnn.py --backbone mobilenet    # Faster R-CNN baseline  (CPU: ~1-2 h; use --backbone resnet50 only on GPU)

# 2. Pick confidence thresholds on VALIDATION, then evaluate once on TEST
python threshold_sweep.py
python evaluate_test.py --crosscheck

# 3. Efficiency (run on the machine you will quote in the paper)
python measure_efficiency.py --threads 4

# 4. Hard cases: init -> open hard_cases/manifest.csv, look at each image, fix has_smoke, set verified=1 -> run
python hard_cases.py init
python hard_cases.py run

# 5. Association rules: dump -> label gt_associated in results/association_candidates.csv -> score
python association_eval.py dump
python association_eval.py score

# 6. Tables + figures
python make_report.py
```

Outputs: `results/*.json`, `results/tables.md`, `results/figures/fig2..fig6*.png`, `runs/*`.
GPU/Colab can be used for steps 1 only (`--device 0` / `--device cuda`); copy `runs/` back and do steps 2-6 locally.

Same steps are also reachable under the names used in the research plan: `ai/training/train_benchmark.py`,
`evaluate_test.py`, `threshold_analysis.py`, `latency_benchmark.py`, `difficult_cases.py`, `generate_results.py`.
Put extra fog/dust/steam/haze images in `ai/dataset/hard_negatives/` before `hard_cases.py init`.
`make_report.py` also copies figures/tables to `paper/` and writes `results/reproducibility.json`.

GPU notes (GTX 1650, 4 GB): `--device 0` (YOLO) / `--device cuda` (others) both work. Use Faster R-CNN `--backbone mobilenet`
(`resnet50` needs `--batch 2` and may still run out of memory). If YOLO loss is NaN or mAP stays 0, add `--no-amp --force`.
If out of memory, rerun ALL YOLO models with the same smaller `--batch` and `--force`.
