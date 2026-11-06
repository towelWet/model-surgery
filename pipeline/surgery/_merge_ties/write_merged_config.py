# write_merged_config.py — copy base (src_a) config to dest, minus exporter-tool bookkeeping keys.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import json
import os

TOOL_KEYS = ("model_name", "unsloth_version", "unsloth_fixed", "unsloth_fixed_mtp")  # describe the fine-tune, not the merge


def write_merged_config(src_a, dest):
    with open(os.path.join(src_a, "config.json")) as f:
        cfg = json.load(f)
    for k in TOOL_KEYS:
        cfg.pop(k, None)
    with open(os.path.join(dest, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
