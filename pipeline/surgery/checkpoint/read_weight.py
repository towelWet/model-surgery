"""Read a float weight or dequantize a complete affine triple, including cross-shard companions."""
import mlx.core as mx
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shard_io import load_shard_cache
from .quantization import quantization_settings


def read_weight(src, key, wmap, cache, cfg):
    weight = load_shard_cache(src, wmap[key], cache)[key]
    prefix = key.removesuffix(".weight")
    companions = [prefix + suffix for suffix in (".scales", ".biases")]
    present = [k in wmap for k in companions]
    if not any(present):
        if not mx.issubdtype(weight.dtype, mx.floating):
            raise ValueError(f"{key}: integer weight has no supported quantization companions")
        return weight.astype(mx.float32), None, weight.dtype
    if not all(present) or weight.dtype != mx.uint32:
        raise ValueError(f"{key}: incomplete or unsupported affine quantization representation")
    options = quantization_settings(cfg, key)
    scales, biases = [load_shard_cache(src, wmap[k], cache)[k] for k in companions]
    if scales.shape != biases.shape or not mx.issubdtype(scales.dtype, mx.floating):
        raise ValueError(f"{key}: invalid affine scales/biases")
    dense = mx.dequantize(weight, scales, biases, **options).astype(mx.float32)
    mx.eval(dense)
    return dense, options, scales.dtype
