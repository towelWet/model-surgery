# _forward.py — qwen3_5 forward with optional layer skip (identity); math only.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# DISABLED 2026-11-06 — fail-closed stub, NOT a port. The legacy body ran an
# in-process mlx_lm teacher-forced forward where skip_layers left h unchanged
# at those indices, then returned normed logits via lm_head / tied embeddings.
# mlx_lm / mlx_vlm are banned, so that body and its mlx imports were removed.
# tensor-array has no layer-skip logits route (checked src/server/route
# 2026-11-06). Do not re-add an in-process forward here; add the route in
# tensor-array.


def forward_logits(model, input_ids, skip_layers=None):
    """Contract (unimplemented): teacher-forced logits [1, seq, vocab] for one packed prompt."""
    raise NotImplementedError(
        "forward_logits() disabled: needs tensor-array teacher-forced logits "
        "instrumentation with layer-skip (block output not applied at skipped "
        "indices). tensor-array src/server/route has no such route (verified "
        "2026-11-06); see pipeline/profile/_mlx_load.py header."
    )
