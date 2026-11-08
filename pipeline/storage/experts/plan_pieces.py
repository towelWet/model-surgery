"""Plan expert-aware files: destack [E,...] tensors into <name>.eNNN rows, assign files via strategy.
NEVER DELETE COMMENTS — explain every non-obvious line.
Update comments with the code; do not strip them.
Output order inside each file: resident tensors, then layer -> expert -> tensor, so every expert's
tensors are one contiguous span (tensor-array src/infer/moe/stream/slab/plan.cpp coalesces that into a single pread).
"""
from pipeline.storage.experts.classify import classify


def plan_pieces(tensors, experts, lay, file_of, group):
    """Return {file: [piece]}; piece has name, source, row (-1 = whole), dtype, shape, full_shape, regions."""
    pieces, names = [], set()
    suffix = lambda e: f".{lay['destacked_prefix']}{e:0{lay['destacked_digits']}d}"
    for order, t in enumerate(tensors):
        info = classify(t["name"], t["shape"], experts, lay)
        base = {"source": t["name"], "dtype": t["dtype"], "full_shape": t["shape"], "order": order,
                "layer": -1 if info["layer"] is None else info["layer"]}
        if info["kind"] != "stacked":
            pieces.append({**base, "name": t["name"], "row": -1, "shape": t["shape"], "info": info,
                           "expert": -1 if info["expert"] is None else info["expert"],
                           "regions": [(t["path"], t["start"], t["nbytes"])]})
            continue
        if t["nbytes"] % experts:
            raise ValueError(f"{t['name']}: {t['nbytes']} bytes not divisible by {experts} experts")
        row = t["nbytes"] // experts                               # row-major: expert e = contiguous byte row
        for e in range(experts):
            pieces.append({**base, "name": t["name"] + suffix(e), "row": e, "shape": t["shape"][1:],
                           "expert": e, "info": {**info, "expert": e},
                           "regions": [(t["path"], t["start"] + e * row, row)]})
    files = {}
    for p in sorted(pieces, key=lambda p: (p["expert"] >= 0, p["layer"], p["expert"], p["order"])):
        if p["name"] in names:
            raise ValueError(f"name collision after destack: {p['name']}")   # e.g. source already had .e007
        names.add(p["name"])
        files.setdefault(file_of(p["info"], group, lay), []).append(p)
    return files
