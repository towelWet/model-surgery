# layer/_core.py — quantize all 7 linears of decoder layer li (RTN / AWQ / GPTQ).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import numpy as np
import mlx.core as mx
from pipeline.surgery.wquant.layer._inputs import layer_inputs
from pipeline.surgery.wquant.awq._fold import awq_fold
from pipeline.surgery.wquant._quant_linear import quant_linear

ROLES = {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj", "o": "self_attn.o_proj",
         "gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj"}


def wquant_layer(lazy, li, files, method, bits, group):
    p = f"model.layers.{li}."
    f32 = lambda k: np.array(lazy[p + k].astype(mx.float32))
    t = {r: f32(n + ".weight") for r, n in ROLES.items()}
    t["ln1"], t["ln2"] = f32("input_layernorm.weight"), f32("post_attention_layernorm.weight")
    xs = layer_inputs(files, li, t["gate"], t["up"])
    if method == "awq":
        t = awq_fold(t, xs, bits, group)
    out = {p + "input_layernorm.weight": mx.array(t["ln1"]).astype(mx.bfloat16),
           p + "post_attention_layernorm.weight": mx.array(t["ln2"]).astype(mx.bfloat16)}
    for r, n in ROLES.items():
        q, s, b = quant_linear(method, t[r], xs[r], bits, group)
        out.update({p + n + ".weight": q, p + n + ".scales": s.astype(mx.bfloat16), p + n + ".biases": b.astype(mx.bfloat16)})
    return out
