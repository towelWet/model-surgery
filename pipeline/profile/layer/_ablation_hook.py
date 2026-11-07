# _ablation_hook.py — JSON body handler for POST /v1/layer_ablation.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# inferserve-model should forward this route to layer_ablation() below.
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))              # profile/ (_mlx_load)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ablation"))  # scorer lives in profile/ablation/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from pipeline.runtime.settings import runtime_settings
from _layer_ablation import layer_ablation               # not `_score`: profile/depth/_score.py shadows it




def handle_layer_ablation(body: dict) -> dict:
    """OpenAI-style extension: prompts + optional zero_layers list."""
    model_dir = runtime_settings()["http"]["model_path"]
    if not model_dir:
        raise RuntimeError("configure the runtime model_path or SURGERY_MODEL")
    prompts = body.get("prompts") or []
    if not prompts:
        raise ValueError("prompts required")
    zero = body.get("zero_layers")
    return layer_ablation(model_dir, prompts, zero_layers=zero)
