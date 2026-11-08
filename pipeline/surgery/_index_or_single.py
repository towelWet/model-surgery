# _index_or_single.py — synthesize a HF weight_map for a lone model.safetensors.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import os

from pipeline.surgery._safetensors_header import safetensors_header

SINGLE = "model.safetensors"                 # HF unsharded filename


def synthesize_index(model_dir):
    """Build a HuggingFace-style weight_map for a single-file checkpoint (Empero)."""
    path = os.path.join(model_dir, SINGLE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"no index.json or {SINGLE} in {model_dir}")
    header = safetensors_header(path)
    weight_map = {k: SINGLE for k in header if k != "__metadata__"}
    meta = header.get("__metadata__") or {}          # tensor-array fuse writes null; consumers dict() it (slerp crash 2026-11-08)
    return {"metadata": meta, "weight_map": weight_map}
