"""Entry point named as in the research plan; the implementation lives in ai/benchmark/make_report.py."""
import runpy
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1] / "benchmark"
sys.path.insert(0, str(BENCH))
sys.argv[0] = str(BENCH / "make_report.py")
runpy.run_path(str(BENCH / "make_report.py"), run_name="__main__")
