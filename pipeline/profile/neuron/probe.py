# probe.py — probe per-layer neuron activations for a prompt set.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Inspired by @lyraaaa neuron-explorer and weight-space programming research.
# Sends prompts through tensor-array API, harvests hidden states per layer,
# reports which neurons activate most for target vs baseline prompts.
#
# Usage: python3 neuron_probe.py --target "prompts.txt" --baseline "base.txt"
#        python3 neuron_probe.py --target "prompts.txt" --layer 29 --top 20
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from _load_lines import load_lines
from _probe_hidden import probe_hidden          # tensor-array probe (was _hidden: profile/_hidden.py shadowed it)

p = argparse.ArgumentParser()
p.add_argument("--target", required=True, help="Target prompts (one per line)")
p.add_argument("--baseline", default=None, help="Baseline prompts (one per line)")
p.add_argument("--layer", type=int, default=None, help="Single layer (default: all)")
p.add_argument("--top", type=int, default=10, help="Top-N neurons to report")
p.add_argument("--out", default=None, help="Output JSON path")
args = p.parse_args()

target_prompts = load_lines(args.target)
base_prompts = load_lines(args.baseline) if args.baseline else []
print(f"Target: {len(target_prompts)}, Baseline: {len(base_prompts)}")

results = {"target": [], "baseline": [], "direction": None}

for label, prompts in [("target", target_prompts), ("baseline", base_prompts)]:
    for prompt in prompts:
        resp = probe_hidden(prompt, args.layer)
        results[label].append(resp)
        print(f"  {label}: {prompt[:60]}...")

if args.out:
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote: {args.out}")
