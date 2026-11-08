# _shared.py — mixed-quant recipe table for this package; data only.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
# recipe -> (low_bits, high_bits). High bits go to sensitive tensors (v_proj, down_proj on
# edge/every-3rd layers, lm_head); everything else gets low bits. Same table as mlx_lm convert.
RECIPE_BITS = {
    "mixed_2_6": (2, 6),
    "mixed_3_4": (3, 4),
    "mixed_3_6": (3, 6),
    "mixed_4_6": (4, 6),
}
QUANT_RECIPES = tuple(RECIPE_BITS)                          # kept: importers and error text use it
