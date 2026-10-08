"""DEPRECATED: the old version only averaged the number of detections per image, which cannot measure
precision/recall/F1 or false positives. Difficult-case evaluation now lives in ai/benchmark/hard_cases.py
(image-level + optional box-level metrics at each model's validation-frozen confidence threshold):

    cd ai/benchmark
    python hard_cases.py init      # build hard_cases/manifest.csv, verify labels by eye
    python hard_cases.py run
"""
import runpy
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1] / "benchmark"
sys.path.insert(0, str(BENCH))
print(__doc__)
if len(sys.argv) > 1 and sys.argv[1] in ("init", "run"):
    sys.argv[0] = str(BENCH / "hard_cases.py")
    runpy.run_path(str(BENCH / "hard_cases.py"), run_name="__main__")
