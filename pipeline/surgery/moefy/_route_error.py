# _route_error.py — how much MLP output a top-K expert choice loses, on held-out tokens.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Compares three choices so the router has a floor and a ceiling:
#   router — top-K of fitted logits (what the converted model will do)
#   oracle — top-K of true expert output norms (best a norm-based router could do)
#   random — K experts at random (what an untrained router would do)
import numpy as np


def route_error(acts, wd, labels, logits, topk, experts, seed=0):
    """Mean relative error |sum_sel O_e - sum_all O_e| / |sum_all O_e| for router/oracle/random."""
    outs = np.stack([acts[:, labels == e] @ wd[:, labels == e].T for e in range(experts)], 1)  # [N, E, H]
    full = outs.sum(1)                                        # dense MLP output
    base = np.linalg.norm(full, axis=1) + 1e-8
    rng = np.random.default_rng(seed)
    picks = {"router": np.argsort(-logits, 1)[:, :topk],
             "oracle": np.argsort(-np.linalg.norm(outs, axis=2), 1)[:, :topk],
             "random": np.argsort(rng.random(logits.shape), 1)[:, :topk]}
    result = {}
    for name, idx in picks.items():
        sel = np.take_along_axis(outs, idx[:, :, None], 1).sum(1)
        result[name] = float((np.linalg.norm(sel - full, axis=1) / base).mean())
    return result
