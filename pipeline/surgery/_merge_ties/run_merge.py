# run_merge.py — full merge: verify configs, merge tensors, write config + single shard + index.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import os
import sys

import mlx.core as mx
import numpy as np

from ._shared import copy_tokenizer, load_index, write_index
from .merge_tensors import merge_tensors
from .verify_configs import verify_configs
from .write_merged_config import write_merged_config


def run_merge(src_a, src_b, dest, density, lam, weight_b, seed, bf16_only):
    verify_configs(src_a, src_b)
    rng = np.random.default_rng(seed)
    idx_a = load_index(src_a)
    idx_b = load_index(src_b)
    wmap_a, wmap_b = idx_a["weight_map"], idx_b["weight_map"]
    common = sorted(set(wmap_a) & set(wmap_b))
    only_a = set(wmap_a) - set(wmap_b)
    only_b = set(wmap_b) - set(wmap_a)
    if only_b:
        sys.exit(f"src_b has {len(only_b)} keys absent from src_a — STOP")
    if only_a:
        print(f"  {len(only_a)} src_a-only keys (e.g. vision) copied unchanged from base")

    copy_tokenizer(src_a, dest)
    write_merged_config(src_a, dest)                       # base config, tool keys dropped

    out = merge_tensors(
        common, only_a, wmap_a, wmap_b, src_a, src_b, {}, {}, None,
        rng, density, weight_b, lam, bf16_only,
    )
    shard_name = "model.safetensors"
    mx.save_safetensors(os.path.join(dest, shard_name), out)
    new_wmap = {k: shard_name for k in out}
    meta = dict(idx_a.get("metadata", {}))
    meta["merge_method"] = "dare_ties"
    meta["merge_density"] = density
    meta["merge_base"] = os.path.basename(src_a)
    meta["merge_other"] = os.path.basename(src_b)
    write_index(dest, meta, new_wmap)
    print(f"\nDone -> {dest}")
    print(f"  method: dare_ties  density={density}  lambda={lam}  weight_b={weight_b}")
    if bf16_only:
        print("  NOTE: bf16 weights — re-quant with: python3 export/mlx/convert.py <dest> <dest-Tau-3bit>")
