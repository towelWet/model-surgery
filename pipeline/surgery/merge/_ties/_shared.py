# _merge_ties/_module.py — TIES+DARE merge math and tensor I/O; no CLI.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Intent: one task vector (src_b - src_a), Bernoulli-pruned and scaled by 1/density.
# This is DARE task arithmetic; it does not implement TIES sign election.
# Quantization is resolved from each checkpoint's actual module metadata.
#
# Reference/mergekit/merge_methods/generalized_task_arithmetic.py (dare_ties recipe).
# _shared.py — module-level settings, path setup and shared imports for this package (was _module.py).
import json
import os
import sys
import mlx.core as mx
import numpy as np
from pipeline.surgery.shard_io import copy_tokenizer, load_index, load_shard_cache, write_index
# Arch keys both sources must agree on (was pinned to one 32L/2560 qwen3_5_text model).
EVAL_EVERY = 10
