# _spec.py — parse a layer-removal spec string into a set of layer indices.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Accepts comma-separated ranges and singles: "4-15,40-43" or "24-39" or "7".
# Non-contiguous specs are the whole point: angular-distance profiling showed
# the cheapest 16-layer removal on Qwen3.8-27B is 4-15 PLUS 40-43 (sum 0.336),
# which beats every contiguous window (best contiguous 28-43 scores 0.459).
# remap_key already handles arbitrary sets, so only the CLI needed widening.
from _format_spec import format_spec          # re-export: callers import both from _spec


def parse_spec(text):
    """Return set of int layer indices described by text."""
    out = set()
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))    # inclusive on both ends
        else:
            out.add(int(part))
    return out
