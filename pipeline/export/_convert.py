# _convert.py — safetensors directory → mixed-bit MLX safetensors (first-party, no mlx_lm).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))                    # export/ (_recipes, _convert_* packages)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "convert"))  # _run / _run_stream moved to export/convert/
from _run import run_convert
from _run_stream import run_convert_stream
from _recipes import DEFAULT_RECIPE, GROUP_SIZE

__all__ = ["run_convert", "run_convert_stream", "DEFAULT_RECIPE", "GROUP_SIZE"]
