# _tag_config.py — copy src config.json to dest, recording which layers had MLP zeroed.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import json
import os


def tag_config(src, dest, zero_layers):
    with open(os.path.join(src, "config.json")) as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)            # VLM nests text config; plain LMs are flat
    tc["mlp_zeroed_layers"] = sorted(zero_layers)  # provenance: loaders/evals can see the surgery
    with open(os.path.join(dest, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
