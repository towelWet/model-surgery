# layer_location_from_weights.py — Index of the layer number segment in dotted weight paths (via down_proj keys).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.


def layer_location_from_weights(weight_keys) -> int:
    """Index of the layer number segment in dotted weight paths (via down_proj keys)."""
    down_keys = sorted(k for k in weight_keys if "down_proj" in k and k.endswith(".weight"))
    if not down_keys:
        raise ValueError("No down_proj weights — cannot infer layer path layout.")
    for layer_location, segment in enumerate(down_keys[0].split(".")):
        if segment.isdigit():
            return layer_location
    raise ValueError(f"Cannot find layer index in down_proj key: {down_keys[0]!r}")
