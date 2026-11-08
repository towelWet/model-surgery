# Patch one neuron-output matrix, then stream all checkpoint tensors to disk.
# Packed integers are always dequantized with explicit source metadata before editing.
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
from pipeline.surgery.shard_io import load_index
from pipeline.surgery.checkpoint.read_weight import read_weight
from pipeline.surgery.checkpoint.write.weight import encode_weight
from pipeline.surgery.checkpoint.edited_groups import edited_groups
from pipeline.surgery.checkpoint.write.groups import write_groups
from pipeline.surgery.checkpoint.destination import validate_destination
from pipeline.surgery._layer_prefix import decoder_layer_prefix
from pipeline.surgery.neuron.patch_columns import patch_columns


def apply_neuron_edit(src, dest, spec, cfg):
    """Validate, edit one logical matrix, and stream a complete output checkpoint."""
    validate_destination(dest, sources=[src])
    idx = load_index(src)
    wmap = idx["weight_map"]
    prefix = decoder_layer_prefix(wmap, cfg.get("text_config", cfg).get("num_hidden_layers"))  # skip vision/mtp stacks
    key = f"{prefix}{spec['layer']}.mlp.{spec['proj']}"
    if key not in wmap:
        raise ValueError(f"target weight not found: {key}")
    dense, options, dtype = read_weight(src, key, wmap, {}, cfg)
    edited = encode_weight(key, patch_columns(dense, spec), options, dtype)
    del dense
    groups = edited_groups(src, wmap, edited)
    write_groups(src, dest, groups, idx, cfg)
    return len(spec["columns"])
