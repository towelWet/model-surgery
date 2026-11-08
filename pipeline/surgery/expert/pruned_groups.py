"""Slice/drop experts in one logical group at a time, retaining no whole-model output dictionary."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
from pipeline.surgery.checkpoint.tensor_groups import tensor_groups
from pipeline.surgery.expert._tensor import expert_tensor


def pruned_groups(source, mapping, plan, n_experts):
    for group in tensor_groups(source, mapping, {}):
        output = {}
        for key, tensor in group.items():
            output.update(expert_tensor(key, tensor, plan, n_experts))
        if output:
            yield output
