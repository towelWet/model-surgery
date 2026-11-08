# _target.py — which checkpoint keys unstructured pruning may touch, and their role.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Only decoder linear weights. Embeddings, lm_head, norms and routers stay dense: they are
# small, and zeroing them breaks token lookup / output / routing far more than it saves.
import re

_PAT = re.compile(r"layers\.(\d+)\.(?:self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$")


def sparse_target(key):
    """(layer, role) with role in q/k/v/o/gate/up/down, or None for keys left untouched."""
    m = _PAT.search(key)
    if not m:
        return None
    return int(m.group(1)), m.group(2) or m.group(3)
