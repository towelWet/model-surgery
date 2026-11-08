# ../../moe/partition/output/layer.py — run one mode on one layer, verify exact coverage, write its npz (never overwrite).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
# ValueError, not assert: validation must survive `python -O` (asserts strip to no-ops).
import os
import numpy as np


def run_layer(stats, dname, name, man, p):
    d = os.path.join(stats, dname)
    mass = np.load(os.path.join(d, "neuron.npy")).astype(np.float64)   # per-neuron mass [W]
    prof = np.abs(np.load(os.path.join(d, "sample.npy")).astype(np.float32))  # [tokens, W]
    shared, experts = (np.asarray(x) for x in man["run"](mass, prof, p))
    if shared.ndim != 1 or shared.shape != (p.shared_width,):
        raise ValueError(f"{dname}/{name}: shared must be rank-1 width {p.shared_width}, got {shared.shape}")
    if experts.shape != (p.experts, p.expert_width):
        raise ValueError(f"{dname}/{name}: experts must be ({p.experts},{p.expert_width}), got {experts.shape}")
    ids = np.concatenate([shared, experts.ravel()])
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError(f"{dname}/{name}: partition ids not integral")
    if not (np.sort(ids) == np.arange(p.width)).all():
        raise ValueError(f"{dname}/{name}: ids must cover 0..{p.width - 1} exactly")
    out = os.path.join(stats, "partition", name, f"{dname}.npz")
    if os.path.exists(out):
        raise ValueError(f"{dname}/{name}: refusing to overwrite {out}")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, shared=shared, experts=experts)
