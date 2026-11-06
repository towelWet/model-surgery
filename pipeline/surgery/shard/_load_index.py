# _load_index.py — load safetensors index from model directory.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import json
import os


def load_index(model_dir):
    """Return parsed model.safetensors.index.json, or synthetic map for lone file."""
    path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from _index_or_single import synthesize_index        # Empero HF dump has no index
    return synthesize_index(model_dir)
