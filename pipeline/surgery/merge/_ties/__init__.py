# _merge_ties — package barrel; one function per sibling file.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from ._shared import *  # package settings + path setup (was _module)
from .mlx_to_np_f32 import mlx_to_np_f32
from .materialize import materialize
from .text_cfg import text_cfg
from .verify_configs import verify_configs
from .is_quantized_key import is_quantized_key
from .dequant_np import dequant_np
from .requant_mx import requant_mx
from .dare_random_np import dare_random_np
from .delta_np import merge_delta_np
from .magnitude_trim_np import magnitude_trim_np
from .combine_np import ties_combine_np
from .multi_np import merge_multi_np
from .run_merge import run_merge
