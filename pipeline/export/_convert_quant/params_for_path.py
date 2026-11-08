# params_for_path.py — Return (group_size, bits) for an already-quantized prefix from config.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.


def quant_params_for_path(
    config: dict,
    path: str,
) -> tuple[int, int]:
    """Return (group_size, bits) for an already-quantized prefix from config."""
    q = config.get("quantization") or config.get("quantization_config") or {}
    if path in q and isinstance(q[path], dict):
        p = q[path]
        return int(p["group_size"]), int(p["bits"])
    return int(q.get("group_size", 64)), int(q.get("bits", 4))
