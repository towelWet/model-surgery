"""Write HF model.safetensors.index.json for an expert-split store (names unique after destack).
NEVER DELETE COMMENTS — explain every non-obvious line.
Update comments with the code; do not strip them.
"""
import json
from pathlib import Path


def write_index(dest, files):
    """files: {file: [piece]}. Standard HF + tensor-array loaders resolve every tensor through this map."""
    total = sum(r[2] for ps in files.values() for p in ps for r in p["regions"])
    weight_map = {p["name"]: f for f, ps in files.items() for p in ps}
    index = {"metadata": {"total_size": total, "format": "moe-split"}, "weight_map": weight_map}
    (Path(dest) / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
