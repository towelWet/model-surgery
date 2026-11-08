"""Core repack logic: convert 1-bit MLX checkpoint to 2-bit.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.

1-bit affine: val = scale * bit + bias  (bit in {0,1})
2-bit affine: val = scale * nib + bias  (nib in {0,1,2,3})
Mapping bit->nib identity preserves numerics exactly (same scales/biases, no rounding).
"""
import json
import os

from pipeline.export.repack._mark_2bit import mark_2bit
from pipeline.export.repack._shards import repack_shards
from pipeline.surgery.shard_io import copy_tokenizer, load_index, write_index
from pipeline.surgery.checkpoint.destination import validate_destination


def repack_model(src: str, dst: str) -> None:
    """Convert entire checkpoint from 1-bit to 2-bit packing."""
    validate_destination(dst, sources=[src])               # Tau name + overlap before mkdir
    with open(os.path.join(src, "config.json")) as f:
        cfg = json.load(f)
    if cfg["quantization"]["bits"] != 1:
        raise ValueError("Source must be 1-bit")
    os.makedirs(dst)
    idx = load_index(src)
    new_wm, done = repack_shards(src, dst, idx["weight_map"])
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(mark_2bit(cfg), f, indent=2)
    write_index(dst, idx.get("metadata", {}), new_wm)
    copy_tokenizer(src, dst)
    print(f"\nDone: {done} weight tensors repacked 1-bit -> 2-bit")
