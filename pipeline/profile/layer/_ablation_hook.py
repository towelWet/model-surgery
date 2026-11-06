# _ablation_hook.py — JSON body handler for POST /v1/layer_ablation.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# inferserve-model should forward this route to layer_ablation() below.
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))              # profile/ (_mlx_load)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ablation"))  # scorer lives in profile/ablation/
from _layer_ablation import layer_ablation               # not `_score`: profile/depth/_score.py shadows it

_MODEL_DIR = os.environ.get("MLX_MAGIC_MODEL", "")


def handle_layer_ablation(body: dict) -> dict:
    """OpenAI-style extension: prompts + optional zero_layers list."""
    if not _MODEL_DIR:
        raise RuntimeError("set MLX_MAGIC_MODEL to loaded checkpoint path")
    prompts = body.get("prompts") or []
    if not prompts:
        raise ValueError("prompts required")
    zero = body.get("zero_layers")
    return layer_ablation(_MODEL_DIR, prompts, zero_layers=zero)
