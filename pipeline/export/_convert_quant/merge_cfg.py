# merge_cfg.py — Merge per-prefix quant entries from a shard partial into base.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.


def merge_quant_cfg(base: dict, partial: dict) -> dict:
    """Merge per-prefix quant entries from a shard partial into base."""
    for key, val in partial.items():
        if key in ("group_size", "bits", "mode"):
            continue
        base[key] = val
    return base
