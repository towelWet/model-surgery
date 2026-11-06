"""Byte offset where tensor payload begins inside a single-tensor safetensors file."""
import struct


def payload_start(tensor_file):
    with open(tensor_file, "rb") as stream:
        return 8 + struct.unpack("<Q", stream.read(8))[0]
