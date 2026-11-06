"""Exclusive 'xb' write of bytes with flush+fsync; refuses existing paths."""
import os


def write_xb(path, data):
    with open(path, "xb") as stream:                         # O_EXCL: never overwrite
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return path
