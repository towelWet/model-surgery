# selection.py — discover L<digits> layer dirs in a stats dir, sorted numerically.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import os, re

from pipeline.surgery._spec import parse_spec

L = re.compile(r"^L(\d+)$")                                  # L02 -> 2; sparse ids allowed


def discover_layers(stats, spec):
    """Return [(id, dirname)] for present L<digits> DIRS; spec must name only present ids."""
    found = {}
    for d in sorted(os.listdir(stats)):
        m = L.match(d)
        if not m or not os.path.isdir(os.path.join(stats, d)):
            continue                                        # files named L<digits> are not layers
        i = int(m.group(1))
        if i in found:
            raise SystemExit(f"moe_partition: duplicate layer id {i}: {found[i]} and {d}")
        found[i] = d
    if spec:
        want = parse_spec(spec)
        missing = sorted(want - set(found))
        if missing:
            raise SystemExit(f"moe_partition: requested layers absent from {stats}: {missing}")
        ids = sorted(want & set(found))
    else:
        ids = sorted(found)
    if not ids:
        raise SystemExit(f"moe_partition: no L<digits> dirs in {stats} matching {spec!r}")
    return [(i, found[i]) for i in ids]
