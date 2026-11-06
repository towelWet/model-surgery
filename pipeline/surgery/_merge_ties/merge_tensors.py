# merge_tensors.py — DARE-TIES merge every shared tensor (quantized triples via fp32), copy base-only keys.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import mlx.core as mx

from ._shared import EVAL_EVERY, load_shard_cache
from .dequant_np import dequant_np
from .is_quantized_key import is_quantized_key
from .materialize import materialize
from .merge_delta_np import merge_delta_np
from .mlx_to_np_f32 import mlx_to_np_f32
from .requant_mx import requant_mx


def merge_tensors(common, only_a, wmap_a, wmap_b, src_a, src_b, cache_a, cache_b, _unused, rng, density, weight_b, lam, bf16_only):
    out = {}
    done = 0
    quant_prefixes = set()
    for key in common:
        if key.endswith(".scales") or key.endswith(".biases"):
            continue
        if is_quantized_key(key, wmap_a):
            prefix = key[: -len(".weight")]
            if prefix in quant_prefixes:
                continue
            quant_prefixes.add(prefix)
            wa = dequant_np(load_shard_cache(src_a, wmap_a[key], cache_a), prefix)
            wb = dequant_np(load_shard_cache(src_b, wmap_b[key], cache_b), prefix)
            merged = merge_delta_np(wa, wb, density, weight_b, lam, rng)
            if bf16_only:
                out[prefix + ".weight"] = mx.array(merged, dtype=mx.bfloat16)
            else:
                w_q, scales, biases = requant_mx(merged)
                out[prefix + ".weight"] = w_q
                out[prefix + ".scales"] = scales
                out[prefix + ".biases"] = biases
        else:
            ta = mlx_to_np_f32(load_shard_cache(src_a, wmap_a[key], cache_a)[key])
            tb = mlx_to_np_f32(load_shard_cache(src_b, wmap_b[key], cache_b)[key])
            merged = merge_delta_np(ta, tb, density, weight_b, lam, rng)
            out[key] = mx.array(merged, dtype=mx.bfloat16 if bf16_only else mx.float32)
        done += 1
        if done % EVAL_EVERY == 0:
            print(f"  merged {done} tensors...")
    for key in sorted(only_a):
        out[key] = materialize(load_shard_cache(src_a, wmap_a[key], cache_a)[key])
    return out
