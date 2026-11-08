# Merge task vectors of N fine-tunes onto base; historical package name kept for CLI compatibility.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import mlx.core as mx
from ._shared import load_shard_cache
from .one import merge_one
from pipeline.surgery.checkpoint.configuration import read_configuration
from pipeline.surgery.checkpoint.read_weight import read_weight
from pipeline.surgery.checkpoint.write.weight import encode_weight


def merge_tensors(sources, maps, only_base, rng, spec, bf16_only):
    seen, caches = set(), [{} for _ in sources]             # one bounded shard cache per source
    configs = [read_configuration(s) for s in sources]
    for key in sorted(maps[0]):                             # base key set drives output
        if key in seen:
            continue
        data = load_shard_cache(sources[0], maps[0][key], caches[0])[key]
        if key.endswith((".scales", ".biases")):
            parent = key.rsplit(".", 1)[0] + ".weight"
            if parent in only_base and not bf16_only:
                seen.add(key)
                yield {key: data}
                continue
            if parent in maps[0]:
                continue                                    # companions are emitted with their packed weight
            raise ValueError(f"orphan quantization companion: {key}")
        if key in only_base:                                # e.g. vision tower absent from fine-tunes
            if bf16_only and key.endswith(".weight"):
                dense, _, _ = read_weight(sources[0], key, maps[0], caches[0], configs[0])
                encoded = encode_weight(key, dense, None, mx.bfloat16)
            else:
                encoded = {key: data}
            seen.update(encoded)
            yield encoded
            continue
        if not key.endswith(".weight") and not mx.issubdtype(data.dtype, mx.floating):
            for s, m, c in zip(sources[1:], maps[1:], caches[1:]):   # integer buffers must agree exactly
                other = load_shard_cache(s, m[key], c)[key]
                if data.shape != other.shape or not bool(mx.array_equal(data, other).item()):
                    raise ValueError(f"nonfloating tensor mismatch: {key}")
            seen.add(key)
            yield {key: data}
            continue
        encoded = merge_one(key, sources, maps, caches, configs, rng, spec, bf16_only)
        seen.update(encoded)
        yield encoded
