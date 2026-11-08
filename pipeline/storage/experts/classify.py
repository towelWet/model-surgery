"""Structurally classify one tensor name: shared, stacked [E,...], or one routed expert; plus layer index.
NEVER DELETE COMMENTS — explain every non-obvious line.
Update comments with the code; do not strip them.
Component walk validated against E (same approach as tensor-array src/infer/moe/stream/expert/key-parse.cpp), never substring search.
"""


def classify(name, shape, experts, lay):
    """Return {kind: shared|stacked|single, expert: int|None, layer: int|None}."""
    parts, layer, kind, expert = name.split("."), None, "shared", None
    is_id = lambda s: s.isdigit() and int(s) < experts             # expert id must exist in config
    stacked_ok = bool(shape) and shape[0] == experts               # leading axis really is experts
    for i, c in enumerate(parts):
        nxt = parts[i + 1] if i + 1 < len(parts) else ""
        if c in lay["layer_components"] and nxt.isdigit() and layer is None:
            layer = int(nxt)                                       # first <layers>.<int> wins
        if c in lay["resident_components"]:
            return {"kind": "shared", "expert": None, "layer": layer}   # shared expert: always resident
        if kind != "shared":
            continue                                               # already decided; keep scanning for layer only
        if c in lay["indexed_components"] and is_id(nxt):
            kind, expert = "single", int(nxt)                      # HF per-expert: experts.<e>.*
        elif (c in lay["stacked_components"] or c in lay["fused_components"]) and not nxt.isdigit():
            kind = "stacked" if stacked_ok else "shared"           # wrong leading dim = not stacked, stay resident
    tail, pre = parts[-1], lay["destacked_prefix"]
    if kind == "shared" and tail.startswith(pre) and is_id(tail[len(pre):]) and len(parts) > 1:
        probe = classify(".".join(parts[:-1]), [experts], experts, lay)   # already-destacked slab tensor
        if probe["kind"] == "stacked":
            kind, expert = "single", int(tail[len(pre):])
    return {"kind": kind, "expert": expert, "layer": layer}
