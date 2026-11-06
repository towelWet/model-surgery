# _apply_edit.py — patch specific neuron output columns in weight matrices.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Weight Space Programming: add scaled direction to neuron output columns.
# No gradients, no training loop — just a directional patch.
# Reference: rikiyat.github.io/blog/weight-space-programming/
import mlx.core as mx

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
from shard_io import load_index, load_shard_cache, copy_tokenizer
from _save_edited import save_edited


def apply_neuron_edit(src, dest, spec, cfg):
    """Patch neuron columns per spec and write new checkpoint."""
    idx = load_index(src)
    wmap = idx["weight_map"]
    shard_cache = {}
    out = {}
    layer_prefix = f"model.layers.{spec['layer']}."        # decoder layer key prefix
    target_key = f"{layer_prefix}mlp.{spec['proj']}"       # full weight key to patch
    patched = 0
    for key, shard in sorted(wmap.items()):
        t = load_shard_cache(src, shard, shard_cache)[key]
        if key == target_key:
            W = t.astype(mx.float32)                       # [out, in] weight matrix
            d = spec["direction"].astype(mx.float32)       # [H] unit direction
            for col in spec["columns"]:
                W = W.at[:, col].add(spec["alpha"] * d)    # add direction to column
                patched += 1
            t = W
        out[key] = t
        if len(out) % 15 == 0:
            mx.eval(*out.values())                         # flush every 15 to cap RAM
    mx.eval(*out.values())
    save_edited(src, dest, cfg, out, wmap, idx)
    return patched
