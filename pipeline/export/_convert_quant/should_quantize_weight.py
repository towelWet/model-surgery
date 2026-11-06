# should_quantize_weight.py — Skip norms, biases-as-weight, and shapes that nn.quantize would reject.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import mlx.core as mx


def should_quantize_weight(prefix: str, weight: mx.array, group_size: int) -> bool:
    """Skip norms, biases-as-weight, and shapes that nn.quantize would reject."""
    if weight.ndim != 2:
        return False
    if weight.shape[-1] % group_size != 0:
        return False
    if any(x in prefix for x in ("norm", "layernorm", "layer_norm", "ln_")):
        return False
    return True
