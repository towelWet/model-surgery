# _group_scores.py — importance of each KV head group from real per-head attention outputs.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Head h's contribution to residual stream = o_in[:, h] @ Wo[:, h]^T. Score = mean L2 norm of
# that contribution over calibration tokens. GQA ties `rep` query heads to one KV head, so a
# KV head can only go together with its query heads: score per group = sum of its heads.
import numpy as np


def group_scores(o_in, wo, n_q, n_kv, head_dim):
    """o_in [N, n_q*D], wo [hidden, n_q*D] -> [n_kv] group importance."""
    rep = n_q // n_kv
    per_head = []
    for h in range(n_q):
        cols = slice(h * head_dim, (h + 1) * head_dim)
        per_head.append(np.linalg.norm(o_in[:, cols] @ wo[:, cols].T, axis=1).mean())
    return np.array(per_head).reshape(n_kv, rep).sum(1)     # q head h belongs to KV group h // rep
