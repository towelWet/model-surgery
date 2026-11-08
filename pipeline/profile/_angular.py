# _angular.py — angular distance between hidden states (PruneMe metric).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import mlx.core as mx


def angular_distance(a, b):
    """Normalized angular distance in [0,1] between two hidden states.

    PruneMe metric: arccos(cos_sim)/pi, averaged over token positions.
    Low value => block barely rotates representation => redundant.
    """
    a = a.astype(mx.float32).reshape(-1, a.shape[-1])    # [S, H]
    b = b.astype(mx.float32).reshape(-1, b.shape[-1])
    an = a / (mx.linalg.norm(a, axis=-1, keepdims=True) + 1e-6)
    bn = b / (mx.linalg.norm(b, axis=-1, keepdims=True) + 1e-6)
    cos = mx.clip(mx.sum(an * bn, axis=-1), -1.0, 1.0)  # [S]
    return float(mx.mean(mx.arccos(cos) / mx.array(3.14159265358979)))
