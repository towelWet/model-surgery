# config.py — derive output metadata before committing a streamed checkpoint.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import json
import os

TOOL_KEYS = ("model_name", "unsloth_version", "unsloth_fixed", "unsloth_fixed_mtp")  # describe the fine-tune, not the merge


def merged_configuration(src_a, bf16_only=False):
    with open(os.path.join(src_a, "config.json")) as f:
        cfg = json.load(f)
    for k in TOOL_KEYS:
        cfg.pop(k, None)
    if bf16_only:
        for section in (cfg, cfg.get("text_config", {})):
            section.pop("quantization", None)
            section.pop("quantization_config", None)
    return cfg
