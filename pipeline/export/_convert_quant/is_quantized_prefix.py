# is_quantized_prefix.py — True when prefix has affine .weight + .scales + .biases siblings.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.


def is_quantized_prefix(prefix: str, weight_map: dict) -> bool:
    """True when prefix has affine .weight + .scales + .biases siblings."""
    if weight_map is None:
        return False
    return (
        prefix + ".scales" in weight_map
        and prefix + ".biases" in weight_map
    )
