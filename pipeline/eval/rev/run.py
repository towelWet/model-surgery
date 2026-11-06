# run.py — reversing eval: x86/ARM decode, binary formats, exploitation, decompiler IR.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Usage: python3 run.py [--only x86,formats] [--out report.toml] [--list]
# Needs tensor-array serving the model under test (profile/_api.py decides URL).
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # eval/ for _eval_paths
from _eval_paths import add_eval_paths

SUITE = add_eval_paths(__file__)
from _run_text_suite import run_text_suite
from _rev_system import REV_SYSTEM

sys.exit(run_text_suite("rev", SUITE, REV_SYSTEM, 200, "reversing eval"))   # 200 tok: answers are terse
