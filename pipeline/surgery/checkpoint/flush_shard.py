"""Commit buffered tensor groups as one output shard; budget bounds both RAM and shard count."""
import mlx.core as mx
from .output_shard import output_shard

SHARD_BYTES = 2 << 30                                       # 2 GiB: ~6 files for 35B-A3B 4-bit, bounded RAM on 32 GB


def flush_shard(dest, pending, mapping, number, occupied):
    if not pending:
        return number                                       # nothing buffered (empty tail)
    shard = output_shard(number + 1, occupied)              # skips names taken by preserved assets
    mx.save_safetensors(str(dest / shard), pending)
    mapping.update({key: shard for key in pending})
    pending.clear()                                         # release buffered tensors before the next group
    return number + 1
