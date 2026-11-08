# _slice_attention.py — drop KV groups (and their query heads) from one layer's attention.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# HF layout [out, in]: q/k/v_proj lose ROWS of dropped heads; o_proj loses COLUMNS of dropped
# query heads. q_norm/k_norm act per head on head_dim and are shared, so they stay as-is.
import mlx.core as mx
from pipeline.surgery.heads._keep_rows import keep_rows


def slice_attention(w, prefix, keep_groups, rep, head_dim, q_rows=None):
    """w: key -> mx array; returns {key: sliced} for q/k/v/o_proj weights of `prefix`.
    q_rows: q_proj rows per query head (2*head_dim when q_proj packs [query | output gate], Qwen3.5)."""
    q_heads = [g * rep + r for g in keep_groups for r in range(rep)]  # query heads of kept groups
    qp = mx.array(keep_rows(q_heads, q_rows or head_dim))   # q_proj rows (gate rows travel with their head)
    qi = mx.array(keep_rows(q_heads, head_dim))             # o_proj input columns: gated output is D wide
    ki = mx.array(keep_rows(keep_groups, head_dim))
    out = {}
    for n, idx, axis in (("q_proj", qp, 0), ("k_proj", ki, 0), ("v_proj", ki, 0), ("o_proj", qi, 1)):
        key = f"{prefix}self_attn.{n}.weight"
        out[key] = mx.take(w[key], idx, axis=axis)
        mx.eval(out[key])
    return out
