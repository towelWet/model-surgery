# is_aligned.py — does a safetensors file start its data section on an 8-byte boundary?
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# mlx.save_safetensors writes no header padding; llama.cpp gguf-py rounds data start up to 8,
# so an unaligned file is read a few bytes off (seen: "mmap length is greater than file size").
import struct


def is_aligned(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]                  # little-endian u64 header length
    return (8 + n) % 8 == 0
