"""Merge one floating tensor or complete affine triple across base + N fine-tunes."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import mlx.core as mx
import numpy as np
from pipeline.surgery.checkpoint.read_weight import read_weight
from pipeline.surgery.checkpoint.write.weight import encode_weight
from .multi_np import merge_multi_np


def merge_one(key, sources, maps, caches, configs, rng, spec, bf16_only):
    tensors = [read_weight(s, key, m, c, cfg) for s, m, c, cfg in zip(sources, maps, caches, configs)]
    base, options, dtype = tensors[0]                       # base quantization decides output encoding
    others = [np.array(t[0]) for t in tensors[1:]]          # dequantized fine-tunes
    merged = merge_multi_np(np.array(base), others, spec["method"], spec["density"],
                            spec["weights"], spec["lambda"], rng,
                            spec.get("gamma", 0.01), spec.get("epsilon", 0.1))  # breadcrumbs / DELLA knobs
    if bf16_only:
        options, dtype = None, mx.bfloat16
    return encode_weight(key, mx.array(merged), options, dtype)
