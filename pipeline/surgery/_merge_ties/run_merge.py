# run_merge.py — full merge: verify configs, merge tensors, write config + single shard + index.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import os
import sys

import mlx.core as mx
import numpy as np

from ._shared import copy_tokenizer, load_index, write_index
from .merge_tensors import merge_tensors
from .config import merged_configuration
from checkpoint.destination import validate_destination
from .merge_preflight import merge_preflight
from checkpoint.write_groups import write_groups


def run_merge(src_a, src_b, dest, density, lam, weight_b, seed, bf16_only):
    validate_destination(dest, sources=[src_a, src_b])
    merge_preflight(src_a, src_b, density, lam, weight_b)
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

    groups = merge_tensors(
        common, only_a, wmap_a, wmap_b, src_a, src_b, {}, {}, None,
        rng, density, weight_b, lam, bf16_only,
    )
    meta = dict(idx_a.get("metadata", {}))
    meta["merge_method"] = "dare_task_arithmetic"
    meta["merge_density"] = density
    meta["merge_base"] = os.path.basename(src_a)
    meta["merge_other"] = os.path.basename(src_b)
    cfg = merged_configuration(src_a, bf16_only)
    write_groups(src_a, dest, groups, {**idx_a, "metadata": meta}, cfg)
    print(f"\nDone -> {dest}")
    print(f"  method: dare_task_arithmetic  density={density}  lambda={lam}  weight_b={weight_b}")
