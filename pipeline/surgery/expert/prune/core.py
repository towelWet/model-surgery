# prune/core.py — REAP-style: rank and remove weakest experts from a MoE checkpoint.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Usage: python3 prune.py <src_dir> <dest_dir> <usage_json> [--remove-frac 0.25] [--top-k K]
# usage_json: profile/expert summary (prune_order) or legacy ranked list; least-used first.
# Top-k is unchanged unless --top-k is given (old code silently did top_k - 1 on every run).
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
from pipeline.surgery.expert.prune._core import prune_experts
from pipeline.surgery.expert._removal_plan import removal_plan
from pipeline.surgery.expert.check_plan import validate_pruning_plan
from pipeline.surgery.shard_io import copy_tokenizer
from pipeline.surgery.checkpoint.destination import validate_destination

p = argparse.ArgumentParser()
p.add_argument("src")
p.add_argument("dest")
p.add_argument("usage_json")
p.add_argument("--remove-frac", type=float, default=0.25)
p.add_argument("--top-k", type=int, default=0, help="new num_experts_per_tok (0 = keep)")
args = p.parse_args()

validate_destination(args.dest, sources=[args.src])        # Tau name + overlap before any writes
with open(args.usage_json) as f:
    usage = json.load(f)
with open(os.path.join(args.src, "config.json")) as f:
    cfg = json.load(f)
tc = cfg.get("text_config", cfg)
E = tc["num_experts"]
plan, new_E = removal_plan(usage, E, args.remove_frac)
validate_pruning_plan(args.src, plan, E)
K = args.top_k or tc.get("num_experts_per_tok")
if type(K) is not int or not 0 < K <= new_E:
    raise ValueError(f"top-k {K} must be an integer in 1..{new_E} remaining experts")
tc["num_experts"], tc["num_experts_per_tok"] = new_E, K
print(f"Pruning {E} -> {new_E} experts/layer on {len(plan)} layers  top-K: {K}")
prune_experts(args.src, args.dest, plan, cfg, E)
print(f"\nDone -> {args.dest}  experts: {E} -> {new_E}")
