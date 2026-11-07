"""DARE merge one floating tensor or complete affine triple."""
import mlx.core as mx
import numpy as np
from checkpoint.read_weight import read_weight
from checkpoint.write_weight import encode_weight
from .merge_delta_np import merge_delta_np


def merge_one(key, sources, maps, caches, configs, rng, density, weight_b, lam, bf16_only):
    tensors = [read_weight(s, key, m, c, cfg) for s, m, c, cfg in zip(sources, maps, caches, configs)]
    a, options, dtype = tensors[0]
    b = tensors[1][0]
    merged = merge_delta_np(np.array(a), np.array(b), density, weight_b, lam, rng)
    if bf16_only:
        options, dtype = None, mx.bfloat16
    return encode_weight(key, mx.array(merged), options, dtype)
