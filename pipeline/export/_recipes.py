# _recipes.py — mixed-bit quantization preset definitions for export/_convert.py.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Per-layer bit assignment via export/_convert_predicate.py (ported from mlx_lm convert).
# Embeddings and lm_head stay at higher bit-width; middle layers can go lower.
# Reference: originally mlx_lm/convert.py mixed_quant_predicate_builder.

RECIPES = {
    "mixed_3_4": {
        # 3-bit middle layers, 4-bit first/last 8 layers + lm_head
        "default_bits":    3,
        "boundary_bits":   4,     # bit-width for first + last N layers
        "boundary_layers": 8,
    },
    "mixed_3_6": {
        # 3-bit middle, 6-bit boundaries — better quality at moderate size cost
        "default_bits":    3,
        "boundary_bits":   6,
        "boundary_layers": 8,
    },
    "mixed_2_6": {
        # Aggressive: 2-bit middle, 6-bit boundaries — maximum compression
        "default_bits":    2,
        "boundary_bits":   6,
        "boundary_layers": 8,
    },
    "mixed_4_6": {
        # Conservative: 4-bit everywhere except 6-bit boundaries — minimal quality loss
        "default_bits":    4,
        "boundary_bits":   6,
        "boundary_layers": 8,
    },
}

DEFAULT_RECIPE = "mixed_3_4"                # best balance for Qwen3.8-48L on M4 32GB
GROUP_SIZE     = 64                         # must match training quantization group_size
