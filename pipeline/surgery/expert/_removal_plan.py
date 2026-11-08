# _removal_plan.py — {layer: set(experts to drop)} from a usage ranking; refuses non-uniform plans.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
# Accepts both formats: profile/expert summary {"per_layer": {li: {"prune_order": [e, ...]}}}
# and legacy {li: [{"expert": e, ...}, ...]}; both are least-used first.
import sys
import math


def removal_plan(usage, n_experts, frac):
    layers = usage.get("per_layer", usage)
    if type(n_experts) is not int or n_experts < 2:
        raise ValueError("pruning requires at least two experts")
    if type(frac) not in (int, float) or not math.isfinite(frac) or not 0 < frac < 1:
        raise ValueError("remove fraction must be finite and strictly between zero and one")
    n_drop = int(n_experts * frac)
    if not 0 < n_drop < n_experts:
        raise ValueError("fraction must remove at least one expert and retain at least one")
    if not isinstance(layers, dict) or not layers:
        raise ValueError("usage must contain nonempty per-layer rankings")
    plan = {}
    for li, ranked in layers.items():
        order = ranked["prune_order"] if isinstance(ranked, dict) else [r["expert"] for r in ranked]
        if any(type(e) is not int for e in order) or sorted(order) != list(range(n_experts)):
            raise ValueError(f"layer {li}: ranking must be a permutation of all expert IDs")
        if str(li) != str(int(li)) or int(li) < 0 or int(li) in plan:
            raise ValueError(f"invalid or duplicate layer ID: {li}")
        plan[int(li)] = set(order[:n_drop])                 # weakest n_drop
    return plan, n_experts - n_drop
