"""Validate merge semantics and parameters before allocating or writing a checkpoint."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import math
from .verify_configs import verify_configs
from .multi_np import METHODS
from ._shared import load_index


def merge_preflight(sources, spec):
    if spec["method"] not in METHODS:
        raise ValueError(f"merge method must be one of {METHODS}")
    if spec["method"] == "ties" and len(sources) < 3:
        raise ValueError("TIES needs at least two fine-tunes; use dare for one")
    if spec["method"] == "slerp" and (len(sources) != 2 or not 0 <= spec["lambda"] <= 1):
        raise ValueError("SLERP needs exactly two models and --lambda (t) in [0, 1]")
    if spec["method"] in ("model_stock", "karcher") and len(sources) < 3:
        raise ValueError(f"{spec['method']} needs at least two fine-tunes")
    if spec["method"] == "nuslerp" and len(sources) != 3:
        raise ValueError("nuslerp needs exactly two fine-tunes")
    values = (spec["density"], spec["lambda"], *spec["weights"])
    if not all(math.isfinite(v) for v in values) or not 0 < spec["density"] <= 1:
        raise ValueError("merge scales must be finite and density must be in (0, 1]")
    if len(spec["weights"]) != len(sources) - 1:
        raise ValueError("need exactly one weight per fine-tune")
    verify_configs(*sources)
    maps = [load_index(src)["weight_map"] for src in sources]
    for other in maps[1:]:
        if set(other) - set(maps[0]):
            raise ValueError("fine-tune has keys absent from base; merge refused")
        for key in set(maps[0]) & set(other):
            if not key.endswith(".weight"):
                continue
            for suffix in (".scales", ".biases"):
                companion = key.removesuffix(".weight") + suffix
                if (companion in maps[0]) != (companion in other):
                    raise ValueError(f"incompatible quantized/dense layouts for {key}")
    return maps
