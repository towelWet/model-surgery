# _partition_loop.py — iterate weight map and partition each MLP layer.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # surgery/ parent

from shard_io import load_shard_cache                    # safetensors cache loader
from partition._core import partition_down_scales, partition_down_weight, partition_gate_up
from partition._down_dense import partition_down_dense


def partition_weights(src, wmap, emit, E, S, H, bits, gsize):
    """Walk sorted weight map, partition MLP weights, emit all keys."""
    shard_cache = {}
    for old_key, shard in sorted(wmap.items()):
        weights = load_shard_cache(src, shard, shard_cache)
        t = weights[old_key]
        m = re.match(
            r"(.*\.layers\.)(\d+)(\.mlp\.)(gate_proj|up_proj|down_proj)(\.weight|\.scales|\.biases)$",
            old_key,
        )
        if not m:                                        # non-MLP key: pass through
            emit(old_key, t)
            continue
        prefix, li, _, proj, suffix = m.groups()
        if proj in ("gate_proj", "up_proj"):
            shared, experts_list = partition_gate_up(t, E, S, H)
        elif bits is None:                               # dense bf16 (Empero)
            shared, experts_list = partition_down_dense(t, E, S, H)
        elif suffix == ".weight":                        # quantized weight
            shared, experts_list = partition_down_weight(t, E, S, H, bits, gsize)
        else:                                            # quantized scales/biases
            shared, experts_list = partition_down_scales(t, E, S, H, gsize)
        emit(f"{prefix}{li}.mlp.shared_expert.{proj}{suffix}", shared)
        for e, et in enumerate(experts_list):
            emit(f"{prefix}{li}.mlp.experts.{e}.{proj}{suffix}", et)
