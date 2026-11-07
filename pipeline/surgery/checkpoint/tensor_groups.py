"""Stream logical tensor groups; affine companions travel with their packed weight."""
from .read_weight import load_shard_cache


def tensor_groups(source, mapping, cache):
    seen = set()
    for key in sorted(mapping, key=lambda k: (mapping[k], k)):
        if key in seen:
            continue
        if key.endswith((".scales", ".biases")):
            if key.rsplit(".", 1)[0] + ".weight" in mapping:
                continue
        keys = [key]
        if key.endswith(".weight"):
            prefix = key.removesuffix(".weight")
            keys += [prefix + suffix for suffix in (".scales", ".biases") if prefix + suffix in mapping]
        group = {name: load_shard_cache(source, mapping[name], cache)[name] for name in keys}
        seen.update(keys)
        yield group
