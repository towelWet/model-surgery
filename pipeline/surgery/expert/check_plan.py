"""Require the pruning plan to cover every actual MoE layer and every expert representation."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from pipeline.checkpoint.tensors import tensor_index
from _expert_tensor import PER_EXPERT, ROW_SLICED


def validate_pruning_plan(source, plan, n_experts):
    expert_layers, routed_layers, individual = set(), set(), {}
    for key, (_, entry, _) in tensor_index(source).items():
        per_expert = PER_EXPERT.fullmatch(key)
        stacked = ROW_SLICED.fullmatch(key)
        if per_expert:
            layer, expert = int(per_expert.group(2)), int(per_expert.group(4))
            expert_layers.add(layer)
            individual.setdefault(layer, set()).add(expert)
        elif stacked:
            layer = int(stacked.group(1))
            routed_layers.add(layer)
            if ".switch_mlp." in key:
                expert_layers.add(layer)
            if not entry["shape"] or entry["shape"][0] != n_experts:
                raise ValueError(f"expert dimension disagrees with config: {key}")
    if not expert_layers or set(plan) != expert_layers or not routed_layers <= set(plan):
        raise ValueError("pruning plan must cover every actual MoE layer; unsupported layouts refused")
    if any(ids != set(range(n_experts)) for ids in individual.values()):
        raise ValueError("checkpoint per-expert IDs must cover the configured expert range")
