# _hidden.py — capture residual-stream hidden state entering each decoder layer.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# DISABLED 2026-11-06 — fail-closed stub, NOT a port. The legacy body ran an
# in-process mlx_lm forward (mirroring mlx_lm/models/qwen3_5.py:254-275, with
# the two-mask split: ssm_mask for linear GatedDeltaNet layers, fa_mask for
# full-attention layers). mlx_lm / mlx_vlm are banned, so that body and its
# mlx imports were removed. tensor-array has no per-layer residual-state route
# (checked src/server/route 2026-11-06) and _mlx_load refuses to load.
# Do not re-add an in-process forward here; add the route in tensor-array.


def capture(model, ids):
    """Contract (unimplemented): states[i] == residual entering layer i, len == n_layers+1."""
    raise NotImplementedError(
        "capture() disabled: needs tensor-array per-layer residual-state "
        "instrumentation (hidden states entering each decoder layer). "
        "tensor-array src/server/route has no such route (verified 2026-11-06); "
        "see pipeline/profile/_mlx_load.py header."
    )
