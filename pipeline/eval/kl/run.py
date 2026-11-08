# run.py — general-quality metric: mean next-token KL(parent || student).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Measures divergence of the student's next-token distribution from the 64L parent
# on ~30 fixed general-text prompts (disjoint from CALIB and CASES). Uses tensor-array
# top-32 logprobs per greedy parent continuation step; KL is computed on the union
# of parent/student top-k supports, each side renormalized (no tail bucket; see _math.py).
# Lower KL = student matches parent general text generation better.
# Parent reference traces are cached once (path: _prompts.CACHE, override MS_KL_CACHE)
# so each student model only needs a single resident server load (no dual-model RAM).
# Usage: python3 run.py --build-cache   (parent served)   then   python3 run.py   (student served)
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
from pipeline.eval._paths import add_eval_paths

add_eval_paths(__file__)                                   # kl/ siblings + profile/_api
import argparse

from pipeline.eval.kl._core import build_parent_cache, eval_student, load_cache
from pipeline.eval._write_report import write_report                     # shared eval/ TOML writer

p = argparse.ArgumentParser(description="KL(parent || student) eval")
p.add_argument("--build-cache", action="store_true", help="record parent traces (serve PARENT first)")
p.add_argument("--out", default="", help="write TOML report (mean_kl) for promotion gate")
p.add_argument("--model", default="served", help="label recorded in report")
a = p.parse_args()
if a.build_cache:
    print("Building parent KL reference cache...")
    build_parent_cache()
    sys.exit(0)
mean_kl = eval_student(load_cache())                       # load_cache refuses if parent cache missing
if a.out:
    write_report(a.out, "kl", a.model, {"mean_kl": float(mean_kl)}, {})
    print(f"report -> {a.out}")
