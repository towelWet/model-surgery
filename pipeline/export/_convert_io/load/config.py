# load/config.py — read checkpoint config.json.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import json
import os


def load_config(model_dir: str) -> dict:
    with open(os.path.join(model_dir, "config.json")) as f:
        return json.load(f)
