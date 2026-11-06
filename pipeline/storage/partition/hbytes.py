"""Raw header file content for one shard: 8-byte length prefix + original JSON bytes."""
import struct
from ..formats.safetensors.header import raw_header
from ..files.member import safe_member


def header_bytes(source, rel):
    raw, _, _ = raw_header(safe_member(source, rel))
    return struct.pack("<Q", len(raw)) + raw
