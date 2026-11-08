# _check_compatible.py — refuse stacking layers from models with different tensor shapes.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Layers only fit together when residual width, attention geometry, MLP width, vocabulary
# and quantization are identical; otherwise shapes mismatch or values mean different things.
_KEYS = ("architectures", "hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim",
         "intermediate_size", "vocab_size", "quantization", "num_experts", "moe_intermediate_size")


def check_compatible(cfgs):
    for k in _KEYS:
        vals = {repr(c.get(k)) for c in cfgs}
        if len(vals) > 1:
            raise ValueError(f"models differ in {k}: {sorted(vals)}")
