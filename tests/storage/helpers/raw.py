"""Raw shard writer for malformed-header cases: 8-byte length prefix + table + payload."""
import struct
from pathlib import Path


def write_shard(path, table, payload):
    """Write one safetensors-shaped blob verbatim; table is already-encoded bytes."""
    Path(path).write_bytes(struct.pack("<Q", len(table)) + table + payload)
