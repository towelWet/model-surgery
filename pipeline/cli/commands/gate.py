# gate.py — `ms gate --base DIR --cand DIR`: promotion gate over eval reports; exit 1 = do not promote.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
# Writes <cand>/verdict.toml so the decision is recorded beside the evidence. Never moves models:
# promoting 2-test/<N> -> 3-final stays a human step.
import argparse
import os
import sys
import tomllib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
from pipeline.cli.gate._check_metric import check_metric
from pipeline.cli.gate._load_reports import load_reports
from pipeline.cli.gate._write_verdict import write_verdict
from pipeline.cli.gate.evidence import check_evidence

RULES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "eval", "gate.toml")


def run(argv):
    p = argparse.ArgumentParser(prog="ms gate")
    p.add_argument("--base", required=True, help="reports dir of base model (ms eval --out-dir)")
    p.add_argument("--cand", required=True, help="reports dir of candidate")
    p.add_argument("--rules", default=RULES)
    a = p.parse_args(argv)
    rules = tomllib.loads(open(a.rules).read())
    base, cand = load_reports(a.base), load_reports(a.cand)
    rows = [check_metric(s, m, r, base.get(s, {}).get(m), cand.get(s, {}).get(m))
            for s, metrics in rules.items() for m, r in metrics.items()]
    if not rows or not any(not r.get("report_only") for ms in rules.values() for r in ms.values()):
        rows.append(("FAIL", "rules need at least one blocking metric"))
    for suite, metrics in rules.items():
        if any(not rule.get("report_only") for rule in metrics.values()):
            rows.extend(check_evidence(suite, base.get(suite, {}), cand.get(suite, {})))
    for status, text in rows:
        print(f"{status:<4} {text}")
    passed = all(s != "FAIL" for s, _ in rows)
    write_verdict(a.cand, passed, rows, a.base)
    print(f"\n{'PROMOTE OK' if passed else 'BLOCKED'} -> {os.path.join(a.cand, 'verdict.toml')}")
    return 0 if passed else 1


COMMAND = {"name": "gate", "help": "promotion gate: candidate eval reports vs base (eval/gate.toml)", "run": run}
