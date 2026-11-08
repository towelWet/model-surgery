# _convert_quant — package barrel; one function per sibling file.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from ._shared import *  # package settings + path setup (was _module)
from .is_quantized_prefix import is_quantized_prefix
from .params_for_path import quant_params_for_path
from .should_quantize_weight import should_quantize_weight
from .dequant_weight import dequant_weight
from .quantize.one import quantize_one
from .quantize.checkpoint_weights import quantize_checkpoint_weights
from .bits_per_weight import bits_per_weight
from .apply_quant_config import apply_quant_config
from .merge_cfg import merge_quant_cfg
from .quantize.prefix import quantize_prefix
