"""Entry point named as in the research plan; the implementation lives in ai/benchmark/hard_cases.py."""
import runpy
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1] / "benchmark"
sys.path.insert(0, str(BENCH))
sys.argv[0] = str(BENCH / "hard_cases.py")
runpy.run_path(str(BENCH / "hard_cases.py"), run_name="__main__")
