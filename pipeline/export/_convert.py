# _convert.py — safetensors directory → mixed-bit MLX safetensors (first-party, no mlx_lm).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
from pipeline.export.convert.run._core import run_convert
from pipeline.export.convert.run._stream import run_convert_stream
from pipeline.export._recipes import DEFAULT_RECIPE, GROUP_SIZE

__all__ = ["run_convert", "run_convert_stream", "DEFAULT_RECIPE", "GROUP_SIZE"]
