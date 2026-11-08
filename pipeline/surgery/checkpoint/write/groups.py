"""Stream tensor groups into budget-sized shards; config commits last, interrupted output stays marked."""
import itertools
import os
import sys
import mlx.core as mx
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")))
from pipeline.storage.partition.inventory import checkpoint_files
from pipeline.surgery.shard._write_index import write_index
from ..destination import validate_destination
from ..ancillary_copy import copy_checkpoint_assets
from ..flush_shard import SHARD_BYTES, flush_shard


def write_groups(source, destination, groups, index, config=None, shard_bytes=SHARD_BYTES):
    dest = validate_destination(destination, sources=[source])
    assets = set(checkpoint_files(source)) - set(index["weight_map"].values())
    marker = ".surgery-incomplete.toml"
    if marker in assets:
        raise ValueError("source is an interrupted surgery checkpoint")
    groups = iter(groups)
    first = next(groups, None)                              # validate first transform before writing
    if first is None:
        raise ValueError("transformation emitted no tensors")
    dest.mkdir()
    (dest / marker).write_text('status = "incomplete"\n')
    mapping, total, occupied, pending, number = {}, 0, set(assets), {}, 0
    stream = itertools.chain((first,), groups)
    del first
    for group in stream:
        if set(group) & (set(mapping) | set(pending)):
            raise ValueError("transformation emitted duplicate tensor names")
        mx.eval(*group.values())
        pending.update(group)                               # groups never split: companions stay together
        total += sum(tensor.nbytes for tensor in group.values())
        del group
        if sum(tensor.nbytes for tensor in pending.values()) >= shard_bytes:
            number = flush_shard(dest, pending, mapping, number, occupied)
    flush_shard(dest, pending, mapping, number, occupied)  # tail shard
    metadata = dict(index.get("metadata", {}))
    metadata["total_size"] = total
    write_index(dest, metadata, mapping)
    copy_checkpoint_assets(source, dest, assets, config)
    (dest / marker).unlink()                                # our own transaction marker, after commit
