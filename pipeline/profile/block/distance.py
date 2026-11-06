# distance.py — REAL layer-importance profiler (PruneMe angular distance).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Replaces block_similarity.py, whose quantized-weight-norm proxy returned an
# identical 1.92e+16 for every group and therefore ranked nothing at all.
# BLOCKED 2026-11-06: needs an tensor-array per-layer residual-state route —
# _hidden.capture() is a fail-closed stub (mlx_lm banned); not runnable today.
#
# Cut candidates align to full_attention_interval so the repeating
# (linear,linear,linear,full) group pattern survives removal intact.
# Several block sizes score from ONE model load, because loading 11 GB
# off the external drive costs ~5 minutes and must not be repeated.
#
# Usage: python3 distance.py <model_dir> [--blocks 4,8,16] [--limit N]
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # profile/
from _mlx_load import load as mlx_load
from calib._core import CALIB
from _sum_distances import sum_distances
from _rank_blocks import rank_blocks

p = argparse.ArgumentParser()
p.add_argument("model")
p.add_argument("--blocks",     default="4,8,16")        # contiguous layer counts to score
p.add_argument("--max-tokens", type=int, default=192)   # truncate prompts; caps peak RAM
p.add_argument("--limit",      type=int, default=0)     # 0 = all; >0 = smoke-test subset
p.add_argument("--out",        default=None)
a = p.parse_args()

BLOCKS  = [int(x) for x in a.blocks.split(",")]
PROMPTS = CALIB[: a.limit] if a.limit else CALIB

cfg      = json.load(open(os.path.join(a.model, "config.json")))
cfg      = cfg.get("text_config", cfg)                 # VLM nests; plain LM is flat
n_layers = cfg["num_hidden_layers"]
stride   = cfg.get("full_attention_interval", 1)        # hybrid attn: keep (lin,lin,lin,full) groups; else any layer
starts   = {b: list(range(0, n_layers - b + 1, stride)) for b in BLOCKS}

print(f"Loading {a.model} ({n_layers} layers) ...")
model, tok = mlx_load(a.model)

totals = sum_distances(model, tok, PROMPTS, starts, a.max_tokens)
result = rank_blocks(a.model, n_layers, len(PROMPTS), totals)

out = a.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "block_distance.json")
with open(out, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nWrote {out}")
