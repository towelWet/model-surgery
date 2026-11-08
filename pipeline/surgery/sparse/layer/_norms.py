# layer/_norms.py — Wanda channel norms for MLP layer owning `key`, from lazily loaded weights.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import numpy as np
import mlx.core as mx
from pipeline.surgery.sparse._channel_norms import channel_norms


def layer_norms(lazy, key, layer, files):
    """Input-channel norms for gate/up/down of `layer`, using its DENSE gate/up weights."""
    tag = f"layers.{layer}."
    p = key[: key.index(tag) + len(tag)] + "mlp."             # same layer's MLP, whatever role `key` has
    wg, wu = (np.array(lazy[p + n + "_proj.weight"].astype(mx.float32)) for n in ("gate", "up"))
    return channel_norms(files, layer, wg, wu)
