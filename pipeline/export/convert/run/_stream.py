# run/_stream.py — shard-at-a-time mixed-bit convert (bf16-safe).
import os
import sys

import mlx.core as mx

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
from pipeline.surgery.checkpoint.destination import validate_destination
from pipeline.export._convert_io import (
    copy_ancillary, copy_tokenizer, load_config, load_index, list_shard_files,
    save_config, save_weights,
)
from pipeline.export._convert_predicate import layer_location_from_weights, mixed_quant_predicate_builder, num_layers_from_config
from pipeline.export._convert_quant import apply_quant_config, bits_per_weight, merge_quant_cfg, quantize_checkpoint_weights
from pipeline.export._recipes import GROUP_SIZE, RECIPES


def run_convert_stream(src, dest, recipe, group_size=GROUP_SIZE):
    validate_destination(dest, sources=[src])              # Tau name + overlap before weight loads
    spec = RECIPES[recipe]
    config = load_config(src)
    shards = list_shard_files(src)
    idx_path = os.path.join(src, "model.safetensors.index.json")
    wmap = load_index(src)["weight_map"] if os.path.exists(idx_path) else None
    sample_keys = list(wmap.keys()) if wmap else list(mx.load(os.path.join(src, shards[0])).keys())
    num_layers = num_layers_from_config(config)
    layer_loc = layer_location_from_weights(sample_keys)
    predicate = mixed_quant_predicate_builder(recipe, num_layers, layer_loc, group_size)
    out_weights, quant_cfg = {}, {
        "group_size": group_size, "bits": spec["default_bits"], "mode": "affine",
    }
    print(f"[INFO] stream convert  recipe={recipe}  shards={len(shards)}")
    for i, sf in enumerate(shards, 1):
        print(f"[INFO] shard {i}/{len(shards)}  {sf}")
        shard = mx.load(os.path.join(src, sf))
        partial, qpart = quantize_checkpoint_weights(
            shard, wmap, config, predicate, group_size, spec["default_bits"],
        )
        out_weights.update(partial)
        merge_quant_cfg(quant_cfg, qpart)
        del shard, partial, qpart
    os.makedirs(dest)
    copy_tokenizer(src, dest)
    save_weights(dest, out_weights)
    save_config(apply_quant_config(config, quant_cfg), dest)
    copy_ancillary(src, dest)
    print(f"[INFO] ~{bits_per_weight(out_weights, quant_cfg):.3f} bpw  ->  {dest}")
