# mixed_bits_for_path.py — quant params for one tensor path under a mixed recipe (llama.cpp Q4_K_M style).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
# Ported from mlx_lm convert's mixed_quant_predicate (logic only; no mlx_lm import).


def mixed_bits_for_path(path, low_bits, high_bits, num_layers, layer_location, group_size=64):
    parts = path.split(".")
    # Empero VLM: merger/patch_embed have no layer index at this slot (e.g. linear_fc1).
    index = 0
    if len(parts) > layer_location and parts[layer_location].isdigit():
        index = int(parts[layer_location])
    use_more_bits = (
        index < num_layers // 8                             # first eighth of layers
        or index >= 7 * num_layers // 8                     # last eighth
        or (index - num_layers // 8) % 3 == 2               # every 3rd layer in the middle
    )
    high = {"group_size": group_size, "bits": high_bits, "mode": "affine"}
    if any(p in path for p in ("v_proj", "v_a_proj", "v_b_proj", "down_proj")) and use_more_bits:
        return high
    if "lm_head" in path:
        return high                                         # output head always high precision
    return {"group_size": group_size, "bits": low_bits, "mode": "affine"}
