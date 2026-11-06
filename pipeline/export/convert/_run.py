# _run.py — mixed-bit convert orchestration; no CLI.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "surgery"))  # checkpoint helpers
from checkpoint.destination import validate_destination
from _convert_io import copy_ancillary, copy_tokenizer, load_all_weights, load_config, save_config, save_weights
from _convert_predicate import layer_location_from_weights, mixed_quant_predicate_builder, num_layers_from_config
from _convert_quant import apply_quant_config, bits_per_weight, quantize_checkpoint_weights
from _recipes import GROUP_SIZE, RECIPES


def run_convert(src, dest, recipe, group_size=GROUP_SIZE):
    validate_destination(dest, sources=[src])              # Tau name + overlap before weight loads
    if recipe not in RECIPES:
        raise ValueError(f"Unknown recipe {recipe!r}; choose from {list(RECIPES)}")
    spec = RECIPES[recipe]
    default_bits = spec["default_bits"]
    print(f"[INFO] Loading weights from {src}")
    config = load_config(src)
    weights, weight_map = load_all_weights(src)
    num_layers = num_layers_from_config(config)
    layer_loc = layer_location_from_weights(weights.keys())
    predicate = mixed_quant_predicate_builder(recipe, num_layers, layer_loc, group_size)
    print(f"[INFO] Quantizing  recipe={recipe}  default_bits={default_bits}  group_size={group_size}")
    out_weights, quant_cfg = quantize_checkpoint_weights(
        weights, weight_map, config, predicate, group_size, default_bits,
    )
    out_config = apply_quant_config(config, quant_cfg)
    os.makedirs(dest)
    copy_tokenizer(src, dest)
    save_weights(dest, out_weights)
    save_config(out_config, dest)
    copy_ancillary(src, dest)
    bpw = bits_per_weight(out_weights, quant_cfg)
    print(f"[INFO] Quantized model with ~{bpw:.3f} bits per weight.")
    print(f"[INFO] Done -> {dest}")
