# Merge DARE task vectors; the historical package name is retained for CLI compatibility.
import mlx.core as mx
from ._shared import load_shard_cache
from .merge_one import merge_one
from checkpoint.configuration import read_configuration
from checkpoint.read_weight import read_weight
from checkpoint.write_weight import encode_weight


def merge_tensors(common, only_a, wmap_a, wmap_b, src_a, src_b, cache_a, cache_b, _unused, rng, density, weight_b, lam, bf16_only):
    seen = set()
    configs = [read_configuration(src_a), read_configuration(src_b)]
    for key in sorted(set(common) | set(only_a)):
        if key in seen:
            continue
        data = load_shard_cache(src_a, wmap_a[key], cache_a)[key]
        if key.endswith((".scales", ".biases")):
            parent = key.rsplit(".", 1)[0] + ".weight"
            if parent in only_a and not bf16_only:
                seen.add(key)
                yield {key: data}
                continue
            if parent in wmap_a:
                continue  # companions are emitted together with their packed weight
            raise ValueError(f"orphan quantization companion: {key}")
        if key in only_a:
            if bf16_only and key.endswith(".weight"):
                dense, _, dtype = read_weight(src_a, key, wmap_a, cache_a, configs[0])
                encoded = encode_weight(key, dense, None, mx.bfloat16)
            else:
                encoded = {key: data}
            seen.update(encoded)
            yield encoded
            continue
        if not key.endswith(".weight") and not mx.issubdtype(data.dtype, mx.floating):
            other = load_shard_cache(src_b, wmap_b[key], cache_b)[key]
            if data.shape != other.shape or not bool(mx.array_equal(data, other).item()):
                raise ValueError(f"nonfloating tensor mismatch: {key}")
            seen.add(key)
            yield {key: data}
            continue
        encoded = merge_one(key, [src_a, src_b], [wmap_a, wmap_b], [cache_a, cache_b],
                            configs, rng, density, weight_b, lam, bf16_only)
        seen.update(encoded)
        yield encoded
