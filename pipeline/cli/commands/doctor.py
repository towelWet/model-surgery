# doctor.py — `ms doctor [check ...]`: run repo health checks from cli/doctor/checks/, exit 1 on problems.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
from pipeline.cli._load_plugins import load_plugins
from pipeline.cli._root import pipeline_root


def run(argv):
    checks = load_plugins(os.path.join(os.path.dirname(__file__), "..", "doctor", "checks"), "CHECK")
    picked = [c for n, c in checks.items() if not argv or n in argv]   # no args = all checks
    bad = 0
    for c in picked:
        problems = c["run"](pipeline_root())
        print(f"{'OK ' if not problems else 'BAD'} {c['name']:<12} {len(problems)}  {c['help']}")
        for line in problems[:40]:                          # cap per check: keep output readable
            print(f"      {line}")
        if len(problems) > 40:
            print(f"      … {len(problems) - 40} more")
        bad += bool(problems) and not c.get("info", False)  # info checks report, never fail
    return 1 if bad else 0


COMMAND = {"name": "doctor", "help": "repo health: imports, bare, size, mlx_lm, litter, docs", "run": run}
