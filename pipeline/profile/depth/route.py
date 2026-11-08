# route.py — Mixture of Depths: score per-layer importance per token.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Inspired by Mixture of Depths (MoD) research and Router-Tuning paper.
# MoE asks which expert; MoD asks which layers are even needed per token.
# Outputs a routing config: which layers can be skipped for which token types.
#
# Uses block ablation internally — for each layer, measures NLL impact of
# skipping it on different prompt categories. High NLL delta = essential layer.
#
# Usage: python3 depth_route.py --prompts corpus.txt [--out route.json]
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
from pipeline.profile._api import _post
from pipeline.profile.depth._score import score_layers

p = argparse.ArgumentParser()
p.add_argument("--prompts", required=True, help="Prompts file (one per line)")
p.add_argument("--out", default="depth_route.json")
p.add_argument("--threshold", type=float, default=0.1, help="NLL delta below which layer is skippable")
args = p.parse_args()

prompts = [l.strip() for l in open(args.prompts) if l.strip()]
print(f"Scoring {len(prompts)} prompts for layer routing...")

route = score_layers(prompts, args.threshold)

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "w") as f:
    json.dump(route, f, indent=2)
print(f"Wrote: {args.out}")
print(f"Skippable layers: {route['skippable']}")
print(f"Essential layers: {route['essential']}")
