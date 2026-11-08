"""Tiny 3-layer MoE GGUF: 4 experts (2 used), per-layer array metadata, distinct values per expert."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import numpy as np
from tests.gguf.helpers.fixture import write_fixture


def moe_fixture(path):
    tensors = {"token_embd.weight": np.arange(64, dtype=np.float32).reshape(2, 32)}
    for layer in range(3):
        base = 1000 * layer
        tensors[f"blk.{layer}.attn_q.weight"] = np.full((32, 32), base, dtype=np.float32)
        tensors[f"blk.{layer}.ffn_gate_inp.weight"] = base + np.arange(4 * 32, dtype=np.float32).reshape(4, 32)
        tensors[f"blk.{layer}.ffn_up_exps.weight"] = base + np.repeat(np.arange(4, dtype=np.float32), 64 * 32).reshape(4, 64, 32)
        tensors[f"blk.{layer}.ffn_up_shexp.weight"] = np.full((64, 32), -1, dtype=np.float32)
    meta = {"block_count": (4, 3), "expert_count": (4, 4), "expert_used_count": (4, 2),
            "attention.head_count": (9, {"type": 4, "items": [10, 11, 12]})}  # per-layer array
    return write_fixture(path, "fixmoe", meta, tensors)
