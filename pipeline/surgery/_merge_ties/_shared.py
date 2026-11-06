# _merge_ties/_module.py — TIES+DARE merge math and tensor I/O; no CLI.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Intent: combine two same-arch fine-tunes without full retrain. Base = src_a;
# task vector = src_b − base. DARE random-prunes the delta (keeps density fraction,
# L1-rescaled). TIES elects sign consensus before summing (matters with ≥2 deltas).
# Result = base + λ · merged_delta. Dequant 4-bit → merge fp32 → requant 4-bit.
#
# Reference/mergekit/merge_methods/generalized_task_arithmetic.py (dare_ties recipe).
# _shared.py — module-level settings, path setup and shared imports for this package (was _module.py).
import json
import os
import sys
import mlx.core as mx
import numpy as np
from shard_io import copy_tokenizer, load_index, load_shard_cache, write_index
# Arch keys both sources must agree on (was pinned to one 32L/2560 qwen3_5_text model).
ARCH_KEYS = ("model_type", "num_hidden_layers", "hidden_size", "vocab_size")
EVAL_EVERY = 10
GROUP_SIZE = 64
BITS = 4
