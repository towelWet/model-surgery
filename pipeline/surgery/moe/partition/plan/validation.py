# ../../moe/partition/plan/validation.py — check partition policy, every input array and every output path up front.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import os
import numpy as np


def validate(stats, items, modes, p):
    """Return {dirname: width}; SystemExit on bad policy, malformed data, or existing output."""
    if p.experts < 1 or p.expert_width < 1 or p.group_size < 1:
        raise SystemExit("experts, expert-width and group-size must be positive")
    if p.shared_width < 0:
        raise SystemExit("shared-width must be >= 0")
    widths = {}
    for _, dname in items:
        d = os.path.join(stats, dname)
        mass = np.load(os.path.join(d, "neuron.npy"), mmap_mode="r")
        samp = np.load(os.path.join(d, "sample.npy"), mmap_mode="r")   # profiles: [tokens, neurons]
        if mass.ndim != 1 or mass.shape[0] == 0:
            raise SystemExit(f"{dname}: neuron.npy must be a non-empty rank-1 array, got {mass.shape}")
        w = mass.shape[0]
        if samp.ndim != 2 or samp.shape[0] == 0 or samp.shape[1] != w:
            raise SystemExit(f"{dname}: sample.npy must be nonempty rank-2 width {w}, got {samp.shape}")
        if not np.isfinite(mass[:]).all() or not np.isfinite(samp[:]).all():
            raise SystemExit(f"{dname}: neuron.npy / sample.npy contain non-finite values")
        if p.experts * p.expert_width + p.shared_width != w:
            raise SystemExit(f"{dname}: experts*expert-width + shared-width != W ({w})")
        if w % p.group_size or p.shared_width % p.group_size or p.expert_width % p.group_size:
            raise SystemExit(f"{dname}: group-size must divide W, shared-width and expert-width")
        for name in modes:
            if not name or name in (".", "..") or "/" in name or "\\" in name:
                raise SystemExit(f"{dname}: unsafe mode name {name!r} (must be one path component)")
            if os.path.exists(os.path.join(stats, "partition", name, f"{dname}.npz")):
                raise SystemExit(f"{dname}: output partition/{name}/{dname}.npz already exists")
        widths[dname] = w
    return widths
