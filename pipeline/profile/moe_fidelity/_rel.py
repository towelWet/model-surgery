# _rel.py — relative output error of masked activations through down_proj.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import mlx.core as mx


def rel(D, a, keep, D_ref=None):
    """Relative output error of D(a*keep) against D_ref(a); D_ref defaults to D."""
    full = a @ (D if D_ref is None else D_ref).T                             # reference output
    masked = (a * keep) @ D.T                                                # masked output
    return float(mx.mean(
        mx.linalg.norm(masked - full, axis=1) /
        (mx.linalg.norm(full, axis=1) + 1e-6)                               # epsilon avoids div-by-zero
    ))
