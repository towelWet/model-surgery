# _n_layers.py — read num_hidden_layers from model config.json.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import json
import os


def n_layers_from_argv(model_dir):
    """Read num_hidden_layers from model's config.json."""
    cfg_path = os.path.join(model_dir, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)
    return tc["num_hidden_layers"]             # no silent default: 48 was one model's depth
