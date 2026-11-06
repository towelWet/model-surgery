# _tap.py — walk every qwen3_5 layer via tap(), handing MLP internals to a callback.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# DISABLED 2026-11-06 — fail-closed stub, NOT a port. The legacy body ran an
# in-process mlx_lm forward (mirroring mlx_lm/models/qwen3_5.py DecoderLayer
# lines 228-240), exposing per layer:
#   m_in — MLP input after post_attention_layernorm   [1, S, 5120]
#   act  — SwiGLU neuron activations silu(gate)*up    [1, S, 17408]
#   out  — layer output; check=True asserted it equals reference layer(h)
# mlx_lm / mlx_vlm are banned, so that body and its mlx imports were removed.
# tensor-array has no MLP tap route (checked src/server/route 2026-11-06).
# Do not re-add an in-process forward here; add the route in tensor-array.


def capture_mlp(model, ids, on_layer, check=False):
    """Contract (unimplemented): call on_layer(index, m_in, act) per layer."""
    raise NotImplementedError(
        "capture_mlp() disabled: needs tensor-array MLP tap instrumentation "
        "(per-layer post_attention_layernorm input and SwiGLU activations). "
        "tensor-array src/server/route has no such route (verified 2026-11-06); "
        "see pipeline/profile/_mlx_load.py header."
    )
