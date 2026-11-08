# ../merge/_ties/text_cfg.py — text config dict of a checkpoint (VLM text_config or flat LM config).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import json
import os


def text_cfg(path):
    with open(os.path.join(path, "config.json")) as f:
        cfg = json.load(f)
    return cfg.get("text_config", cfg)
