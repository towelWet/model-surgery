# materialize.py — Eval and detach from MLX graph — prevents Metal timeout on long converts.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import mlx.core as mx


def materialize(t):
    """Eval and detach from MLX graph — prevents Metal timeout on long converts."""
    mx.eval(t)
    return t
