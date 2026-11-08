# _core.py — re-export all partition functions from split files.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Qwen3.8 partition: intermediate_size 17408 = E*S + H_shared
# Default: E=64, S=192, H=5120 → top-4 active = 4*192+5120 = 5888 intermediate/token.
# Empero Qwen3.5-9B: 12288 = 64*128 + 4096 → top-4 active = 4*128+4096 = 4608.
from ._gate_up import partition_gate_up                   # gate_proj / up_proj slicer
from .down._weight import partition_down_weight            # quantized weight slicer
from .down._scales import partition_down_scales            # scales / biases slicer
