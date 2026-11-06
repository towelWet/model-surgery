# run.py — code eval: Python, frontend and code-review tasks graded by required terms.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Usage: python3 run.py [--only python,review] [--out report.toml] [--list]
# Needs tensor-array serving the model under test (profile/_api.py decides URL).
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # eval/ for _eval_paths
from _eval_paths import add_eval_paths

SUITE = add_eval_paths(__file__)
from _run_text_suite import run_text_suite
from _code_system import CODE_SYSTEM

sys.exit(run_text_suite("code", SUITE, CODE_SYSTEM, 300, "code eval"))   # 300 tok: enough for short functions
