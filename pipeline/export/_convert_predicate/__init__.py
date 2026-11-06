# _convert_predicate — package barrel; one function per sibling file.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from ._shared import *  # package settings + path setup (was _module)
from .num_layers_from_config import num_layers_from_config
from .layer_location_from_weights import layer_location_from_weights
from .mixed_quant_predicate_builder import mixed_quant_predicate_builder
