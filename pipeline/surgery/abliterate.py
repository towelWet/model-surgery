# abliterate.py — project refusal direction out of attn out_proj + down_proj weights.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Usage: python3 abliterate.py <src_dir> <dest_dir> [--alpha 1.0]
import argparse
import json
import os
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(__file__) + "/../profile")
from _embed import embed

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "abliterate_parts"))  # _core / _data live here
from _run_abliterate import run_abliterate
from _data import HARMFUL, HARMLESS
from checkpoint.destination import validate_destination

p = argparse.ArgumentParser()
p.add_argument("src")
p.add_argument("dest")
p.add_argument("--alpha", type=float, default=1.0)
args = p.parse_args()

validate_destination(args.dest, sources=[args.src])        # before mkdir and embed calls

print("Probing refusal direction via API embeddings...")
harmful_vecs = np.stack([embed(pr) for pr in HARMFUL])
harmless_vecs = np.stack([embed(pr) for pr in HARMLESS])
r = harmful_vecs.mean(0) - harmless_vecs.mean(0)
r_hat = mx.array(r / (np.linalg.norm(r) + 1e-9), dtype=mx.float32)
print(f"Direction norm: {np.linalg.norm(r):.4f}")

with open(os.path.join(args.src, "config.json")) as f:
    cfg = json.load(f)
run_abliterate(args.src, args.dest, r_hat, args.alpha, cfg)
print(f"\nDone -> {args.dest}  alpha={args.alpha}")
