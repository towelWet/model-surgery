# prune/_key.py — prune one target tensor by chosen method; caches one layer's calibration stats.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# magnitude: |W|; wanda: |W| x input norm; sparsegpt: OBS mask + error compensation (changes
# kept weights too). ctx: method, sparsity, nm, files, cache {"layer", "stats"} (mutated).
import numpy as np
import mlx.core as mx
from pipeline.surgery.sparse.prune._weight import prune_weight
from pipeline.surgery.sparse.layer._norms import layer_norms
from pipeline.surgery.sparse.layer._hessians import layer_hessians
from pipeline.surgery.sparse._sparsegpt_matrix import sparsegpt_matrix


def prune_key(key, w, tgt, lazy, ctx):
    layer, role = tgt
    if ctx["method"] != "magnitude" and ctx["cache"].get("layer") != layer:
        stats = layer_hessians if ctx["method"] == "sparsegpt" else layer_norms
        ctx["cache"] = {"layer": layer, "stats": stats(lazy, key, layer, ctx["files"])}
    if ctx["method"] == "sparsegpt":
        p = sparsegpt_matrix(np.array(w.astype(mx.float32)), ctx["cache"]["stats"][role], ctx["sparsity"], ctx["nm"])
        out = mx.array(p).astype(w.dtype)
        mx.eval(out)
        return out, float((p == 0).mean())
    norms = ctx["cache"]["stats"][role] if ctx["method"] == "wanda" else None
    return prune_weight(w, ctx["sparsity"], norms, ctx["nm"])
