# _axis.py — which axis of a tensor indexes residual (hidden) channels, or None.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# HF [out, in] layout: readers of residual (q/k/v/gate/up, embeddings, lm_head) have hidden
# on axis 1; writers into residual (o_proj, down_proj) on axis 0; RMSNorm weights are [H].
# q_norm/k_norm act on head_dim, not hidden, so they are never sliced.
_READ = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "embed_tokens", "lm_head")
_WRITE = ("o_proj", "down_proj")


def hidden_axis(key, shape, hidden):
    if "q_norm" in key or "k_norm" in key:
        return None
    if any(n in key for n in _WRITE):
        return 0
    if any(n in key for n in _READ):
        return 1
    if key.endswith("norm.weight") and tuple(shape) == (hidden,):
        return 0                                              # input/post-attention/final RMSNorm
    return None
