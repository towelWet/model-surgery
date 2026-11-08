# ../moefy/hybrid/shared/_tensors.py — shared expert + its sigmoid gate for one layer (Qwen3.5-MoE layout).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Runtime multiplies shared output by sigmoid(x @ shared_expert_gate.T). Gate weight = 0 gives
# exactly 0.5 for every token, so down_proj is doubled: shared neurons then add up like dense ones.
# Gate stays bf16 (qlinear's unquantized path); quantizing an all-zero row has no scale to store.
import mlx.core as mx
from pipeline.surgery.moefy._quant_pack import quant_pack


def shared_tensors(p, wg, wu, wd, ids, group, bits):
    """p = "...layers.{li}.mlp."; ids = shared neuron ids -> dict of mx arrays."""
    out = {}
    out.update(quant_pack(p + "shared_expert.gate_proj", wg[ids], group, bits))
    out.update(quant_pack(p + "shared_expert.up_proj", wu[ids], group, bits))
    out.update(quant_pack(p + "shared_expert.down_proj", wd[:, ids] * 2.0, group, bits))  # undo sigmoid(0)
    out[p + "shared_expert_gate.weight"] = mx.zeros((1, wg.shape[1]), dtype=mx.bfloat16)
    return out
