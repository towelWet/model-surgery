# mixed_quant_predicate_builder.py — path -> quant-params callable for a named mixed recipe.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import functools

from ._shared import QUANT_RECIPES, RECIPE_BITS
from .mixed_bits_for_path import mixed_bits_for_path


def mixed_quant_predicate_builder(recipe, num_layers, layer_location, group_size=64):
    if recipe not in RECIPE_BITS:
        raise ValueError(f"Invalid quant recipe {recipe!r}; choose from {QUANT_RECIPES}")
    low_bits, high_bits = RECIPE_BITS[recipe]
    return functools.partial(mixed_bits_for_path, low_bits=low_bits, high_bits=high_bits,
                             num_layers=num_layers, layer_location=layer_location,
                             group_size=group_size)         # partial, not closure: one function per file
