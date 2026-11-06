# run_all.py — run every eval suite (eval/*/run.py) in sequence, one subprocess each.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Replaces profile/run_master_bench.py (called tool_eval.py/code_eval.py/rev_eval.py,
# files that no longer existed). Suites are discovered: add eval/<name>/run.py = new suite.
# Subprocess per suite: suites reuse module names (_system, cases_*) that would clash in one process.
# Usage: python3 run_all.py [--out-dir DIR] [--skip kl,tool]
import argparse
import glob
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
p = argparse.ArgumentParser(description="run all eval suites")
p.add_argument("--out-dir", default="", help="write <suite>.toml reports here (gate input)")
p.add_argument("--skip", default="kl", help="comma list; kl skipped by default (needs parent cache)")
p.add_argument("--model", default="served", help="label written into every report")
a = p.parse_args()
skip = set(filter(None, a.skip.split(",")))

rc = 0
for run in sorted(glob.glob(os.path.join(HERE, "*", "run.py"))):
    suite = os.path.basename(os.path.dirname(run))
    if suite in skip or suite.startswith("_"):
        continue
    cmd = [sys.executable, run]
    if a.out_dir:                                          # every suite accepts --out <toml>
        cmd += ["--out", os.path.join(a.out_dir, f"{suite}.toml"), "--model", a.model]
    print(f"\n=== {suite} ===", flush=True)
    rc |= subprocess.call(cmd)
sys.exit(rc)
