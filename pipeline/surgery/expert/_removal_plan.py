# _removal_plan.py — {layer: set(experts to drop)} from a usage ranking; refuses non-uniform plans.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
# Accepts both formats: profile/expert summary {"per_layer": {li: {"prune_order": [e, ...]}}}
# and legacy {li: [{"expert": e, ...}, ...]}; both are least-used first.
import sys


def removal_plan(usage, n_experts, frac):
    layers = usage.get("per_layer", usage)
    n_drop = max(1, int(n_experts * frac))                  # same count every layer: config has ONE num_experts
    plan = {}
    for li, ranked in layers.items():
        order = ranked["prune_order"] if isinstance(ranked, dict) else [r["expert"] for r in ranked]
        if len(order) != n_experts:
            sys.exit(f"layer {li}: ranking covers {len(order)} experts, config says {n_experts} — STOP")
        plan[int(li)] = set(order[:n_drop])                 # weakest n_drop
    return plan, n_experts - n_drop
