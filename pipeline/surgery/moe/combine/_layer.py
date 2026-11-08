# ../moe/combine/_layer.py — one MoE layer whose experts are whole MLPs of separate dense fine-tunes.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Unlike moefy (one MLP split into narrow experts, outputs x topk), each expert here is a full
# MLP and router weights already sum to 1 over top-K, so no output rescale.
import numpy as np
import mlx.core as mx
from pipeline.surgery.moefy._quant_pack import quant_pack


def combine_layer(li, expert_weights, router, group, bits):
    p = f"model.layers.{li}.mlp."
    stack = lambda n: np.stack([np.array(w[p + n + ".weight"].astype(mx.float32)) for w in expert_weights])
    out = {}
    for n in ("gate_proj", "up_proj", "down_proj"):
        out.update(quant_pack(p + "switch_mlp." + n, stack(n), group, bits))   # [E, out, in]
    out.update(quant_pack(p + "gate", router, group, bits))  # [E, H]: logits = x @ gate.T
    return out
