# _mark_2bit.py — config quantization block: default + every per-prefix 1-bit entry -> 2-bit.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
# Old code changed only the default; per-prefix {"bits": 1} overrides would have stayed 1 and
# made loaders unpack 2-bit tensors as 1-bit.


def mark_2bit(cfg):
    for key in ("quantization", "quantization_config"):   # mlx writes both; keep them in sync
        q = cfg.get(key)
        if not isinstance(q, dict):
            continue
        q["bits"] = 2
        q["_repacked_from"] = "1-bit affine"
        for v in q.values():
            if isinstance(v, dict) and v.get("bits") == 1:
                v["bits"] = 2
    return cfg
