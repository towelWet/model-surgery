# quantize_checkpoint_weights.py — quantize all eligible .weight tensors; return (out_weights, quant_config).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
# predicate receives the prefix path (no .weight suffix), same as mlx_lm nn paths.
from typing import Callable, Optional, Union

import mlx.core as mx

from ._shared import EVAL_EVERY, materialize
from .quantize_prefix import quantize_prefix


def quantize_checkpoint_weights(
    weights: dict,
    weight_map: Optional[dict],
    config: dict,
    predicate: Callable[[str], Union[bool, dict]],
    default_group_size: int,
    default_bits: int,
) -> tuple[dict, dict]:
    out = {}
    quant_cfg = {"group_size": default_group_size, "bits": default_bits, "mode": "affine"}  # defaults
    done = 0
    for key in sorted(weights.keys()):
        if key.endswith(".scales") or key.endswith(".biases"):
            continue                                        # handled with their .weight prefix
        if not key.endswith(".weight"):
            out[key] = materialize(weights[key])            # non-weight tensors pass through
        else:
            prefix = key[: -len(".weight")]
            tensors, entry = quantize_prefix(weights, weight_map, config, predicate,
                                             prefix, default_group_size, default_bits)
            out.update(tensors)
            if entry:
                quant_cfg[prefix] = entry
        done += 1
        if done % EVAL_EVERY == 0:
            mx.eval(list(out.values())[-3:])                # bound lazy graph growth
    return out, quant_cfg
