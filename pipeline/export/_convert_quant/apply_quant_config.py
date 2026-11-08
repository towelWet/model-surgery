# apply_quant_config.py — Return config copy with quantization + quantization_config set.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import copy


def apply_quant_config(config: dict, quant_cfg: dict) -> dict:
    """Return config copy with quantization + quantization_config set."""
    cfg = copy.deepcopy(config)
    cfg["quantization"] = quant_cfg
    cfg["quantization_config"] = quant_cfg
    return cfg
