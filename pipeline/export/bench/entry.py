# entry.py — post-export bench: rough speed + tool-call accuracy; diff vs baseline JSON.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Usage: python3 entry.py [--baseline bench_baseline.json] [--out bench_result.json] [--max-cases N]
# Needs tensor-array serving exported model. Speed numbers are approximate (see _bench_speed.py).
import argparse
import json
import os
import sys

_PIPE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # pipeline/
sys.path.insert(0, os.path.join(_PIPE, "eval"))          # _eval_paths + _load_cases
from _eval_paths import add_eval_paths

add_eval_paths(os.path.join(_PIPE, "eval", "tool", "run.py"))   # same paths tool eval uses
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # bench/ siblings first
from _load_cases import load_cases
from _tool import tool_metrics
from _bench_speed import measure_speed
from _bench_delta import print_delta

CASES, _ = load_cases(os.path.join(_PIPE, "eval", "tool"))      # was _schema.CASES (numbered shards)

p = argparse.ArgumentParser()
p.add_argument("--baseline", help="Prior bench JSON to diff against")
p.add_argument("--out", default="bench_result.json")
p.add_argument("--max-cases", type=int, default=0, help="0 = all CASES")
args = p.parse_args()

cases = CASES[: args.max_cases] if args.max_cases else CASES
result = measure_speed()
print("Running tool-call eval...")
result.update(tool_metrics(cases))
print(f"\n{json.dumps(result, indent=2)}")
print_delta(result, args.baseline)

with open(args.out, "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved: {args.out}")
