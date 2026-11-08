# _parse_slices.py — parse "0:0-15,1:8-23" into ordered (source, layer) pairs.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Source index = position of model in CLI list; ranges are inclusive, order is output order.


def parse_slices(spec, layer_counts):
    """Return [(source, layer), ...]; every layer must exist in its source."""
    out = []
    for part in spec.split(","):
        src, rng = part.strip().split(":")
        lo, hi = (int(v) for v in rng.split("-")) if "-" in rng else (int(rng), int(rng))
        s = int(src)
        if not 0 <= s < len(layer_counts) or not 0 <= lo <= hi < layer_counts[s]:
            raise ValueError(f"slice {part!r} outside source {s} (layers 0..{layer_counts[s] - 1 if 0 <= s < len(layer_counts) else '?'})")
        out += [(s, l) for l in range(lo, hi + 1)]
    if not out:
        raise ValueError("empty slice spec")
    return out
