# _layer_tensors.py — stacked Qwen3MoE tensors for one layer from dense MLP + fitted router.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Layout = what runtime qwen3/moe reads: mlp.gate [E, H]; switch_mlp.{gate,up}_proj [E, I/E, H];
# switch_mlp.down_proj [E, H, I/E]. Runtime weights each chosen expert by renormalised softmax
# (~1/K when logits are tiny), so down_proj is multiplied by K: selected experts then add up
# like dense neurons. With K = E the layer equals dense MLP up to that ~logit_std weight wobble.
import numpy as np
from pipeline.surgery.moefy._quant_pack import quant_pack


def layer_tensors(li, wg, wu, wd, labels, router, topk, group, bits, prefix="model.layers.", block="mlp"):
    """dict of quantized mx arrays for layer li; neurons grouped by labels into experts."""
    experts = router.shape[1]
    order = np.argsort(labels, kind="stable")                 # expert 0's neurons first, then 1, ...
    width = len(labels) // experts
    p = f"{prefix}{li}.{block}."                               # block: "mlp" (Qwen3-MoE) or "block_sparse_moe" (Mixtral)
    out = {}
    out.update(quant_pack(p + "switch_mlp.gate_proj", wg[order].reshape(experts, width, -1), group, bits))
    out.update(quant_pack(p + "switch_mlp.up_proj", wu[order].reshape(experts, width, -1), group, bits))
    down = wd[:, order].reshape(wd.shape[0], experts, width).transpose(1, 0, 2) * topk  # [E, H, I/E]
    out.update(quant_pack(p + "switch_mlp.down_proj", down, group, bits))
    out.update(quant_pack(p + "gate", router.T, group, bits))  # [E, H]: logits = x @ gate.T
    return out
