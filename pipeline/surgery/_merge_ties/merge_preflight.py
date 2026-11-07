"""Validate merge semantics and parameters before allocating or writing a checkpoint."""
import math
from .verify_configs import verify_configs
from ._shared import load_index


def merge_preflight(src_a, src_b, density, lam, weight_b):
    if not all(math.isfinite(v) for v in (density, lam, weight_b)) or not 0 < density <= 1:
        raise ValueError("merge scales must be finite and density must be in (0, 1]")
    verify_configs(src_a, src_b)
    maps = [load_index(src)["weight_map"] for src in (src_a, src_b)]
    for key in set(maps[0]) & set(maps[1]):
        if not key.endswith(".weight"):
            continue
        for suffix in (".scales", ".biases"):
            companion = key.removesuffix(".weight") + suffix
            if (companion in maps[0]) != (companion in maps[1]):
                raise ValueError(f"incompatible quantized/dense layouts for {key}")
