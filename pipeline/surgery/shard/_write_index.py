# _write_index.py — write safetensors index JSON to destination.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import json
import os


def write_index(dest_dir, metadata, weight_map):
    """Write model.safetensors.index.json to dest_dir."""
    path = os.path.join(dest_dir, "model.safetensors.index.json")
    with open(path, "w") as f:
        json.dump(
            {"metadata": metadata, "weight_map": weight_map},
            f, indent=2,
        )
