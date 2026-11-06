"""Read one safetensors raw header (8-byte length + JSON), duplicate keys rejected."""
import json
import struct
from .duplicates import strict_object


def raw_header(path, max_header=100_000_000):
    """Return (raw_header_bytes, table, data_size) for one shard file."""
    with open(path, "rb") as stream:
        total = stream.seek(0, 2)
        stream.seek(0)
        if total < 8:
            raise ValueError(f"{path}: truncated safetensors prefix")
        length = struct.unpack("<Q", stream.read(8))[0]
        if length > total - 8 or length > max_header:
            raise ValueError(f"{path}: invalid or oversized safetensors header")
        raw = stream.read(length)
    table = json.loads(raw, object_pairs_hook=strict_object)
    if not isinstance(table, dict):
        raise ValueError(f"{path}: safetensors header is not an object")
    return raw, table, total - 8 - length
