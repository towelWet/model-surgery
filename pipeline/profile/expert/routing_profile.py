#!/usr/bin/env python3
# routing_profile.py — real expert routing profiler for MoE surgery, via tensor-array.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# tensor-array records routing itself (TARR_HOPE_STATS); this script only feeds prompts and
# reads the counts back — no in-process forward, no mlx_lm (banned). Serve the model first:
#   TARR_EMBEDDINGS=1 TARR_HOPE_STATS=<stats> inferserve-model ...   (calibration only: slow)
# Usage: python3 routing_profile.py <model_dir> --stats <stats> [--limit N] [--out F]
# /v1/embeddings = one pure forward per prompt (no chat template, no decode tokens).
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from calib._core import CALIB
from _embed import embed                                   # profile/: POST /v1/embeddings
from _read_hope_counts import read_hope_counts
from _summarize import summarize_routing

p = argparse.ArgumentParser()
p.add_argument("model"); p.add_argument("--limit", type=int, default=0)
p.add_argument("--stats", required=True, help="same path as server's TARR_HOPE_STATS")
p.add_argument("--out", default=None)
a = p.parse_args()
PROMPTS = CALIB[:a.limit] if a.limit else CALIB

with open(os.path.join(a.model, "config.json")) as f:
    tc = json.load(f).get("text_config", {})
n_layers = tc["num_hidden_layers"]
n_experts, top_k = tc["num_experts"], tc["num_experts_per_tok"]
print(f"Profiling: {n_layers}L x {n_experts}E top-{top_k}")

h0, c0 = read_hope_counts(a.stats, n_layers, n_experts)   # server table is cumulative: baseline
for pi, prompt in enumerate(PROMPTS, 1):
    embed(prompt)                                          # vector unused; forward fills stats file
    print(f"  [{pi}/{len(PROMPTS)}] {prompt[:50]}")
h1, c1 = read_hope_counts(a.stats, n_layers, n_experts)
hits, coact = h1 - h0, c1 - c0                             # only this run's prompts
moe = hits.sum(axis=1)                                     # routed slots per layer
if not moe.any():
    sys.exit(f"no routing recorded in {a.stats}: server not started with TARR_HOPE_STATS?")
total_tok = int(moe[moe > 0][0] // top_k)                  # each token fills top_k slots per layer

out = a.out or os.path.join(os.path.dirname(__file__), "expert_usage.json")
result = summarize_routing(a.model, n_layers, n_experts, top_k, total_tok,
                           len(PROMPTS), hits, coact)
with open(out, "w") as f:
    json.dump(result, f, indent=2)
nu = sum(r["never_used"] for r in result["per_layer"].values())
nm = sum(len(r["merge_candidates"]) for r in result["per_layer"].values())
print(f"\nTokens={total_tok} never_used={nu} co-fire(j>0.7)={nm}")
print(f"Wrote {out}")
