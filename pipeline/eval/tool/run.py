# run.py — tool eval: does model pick right tool with valid JSON args?
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Usage: python3 run.py [--only bash,no_tool] [--out report.toml] [--list]
# Needs tensor-array serving the model under test (profile/_api.py decides URL).
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # eval/ for _eval_paths
from _eval_paths import add_eval_paths

SUITE = add_eval_paths(__file__)
from _load_cases import load_cases
from _suite_args import suite_args
from _write_report import write_report

a = suite_args("tool-call eval")
cases, by_cat = load_cases(SUITE, a.only)
if a.list:                                                       # dry inspection: no server needed
    for cat, cs in by_cat.items():
        print(f"{cat:14} {len(cs)}")
    sys.exit(0)

from _tool_loop import run_cases                                 # late import: --list needs no _chat
per, sel, arg, scored = {}, 0, 0, 0
for cat, cs in by_cat.items():                                   # score per category for report
    print(f"\n[{cat}]")
    s, g, n = run_cases(cs)
    per[cat] = {"selected": s, "total": len(cs), "args_ok": g, "args_scored": n}
    sel, arg, scored = sel + s, arg + g, scored + n
total = len(cases)
print(f"\ntool_selection_acc : {sel}/{total} = {sel / total:.1%}")
print(f"arg_validity       : {arg}/{scored} = {arg / scored:.1%}" if scored else "arg_validity       : n/a")
if a.out:
    m = {"selected": sel, "total": total, "acc": sel / total, "arg_validity": arg / scored if scored else 0.0}
    write_report(a.out, "tool", a.model, m, per)
    print(f"report -> {a.out}")
