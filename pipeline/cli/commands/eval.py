# eval.py — `ms eval [run_all args]`: run every eval suite against the served model (eval/run/all.py).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
# Example: ms eval --out-dir db/<family>/<ver>/<size>/<N>/2-test/<k>/eval --model <label>
import os
import subprocess
import sys

RUN_ALL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "eval", "run/all.py")


def run(argv):
    return subprocess.call([sys.executable, RUN_ALL] + argv)  # args pass straight through


COMMAND = {"name": "eval", "help": "run eval suites (tool/code/rev[/kl]) -> TOML reports", "run": run}
