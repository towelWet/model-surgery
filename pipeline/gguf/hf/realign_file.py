# realign_file.py — copy safetensors with header space-padded so data starts on 8-byte boundary.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Safetensors spec allows trailing spaces in header JSON; tensor offsets are relative to data
# start, so only header length changes. Data bytes are streamed unchanged (no dtype handling).
import shutil
import struct


def realign_file(src, dst):
    with open(src, "rb") as f, open(dst, "xb") as g:          # "x": never overwrite
        n = struct.unpack("<Q", f.read(8))[0]
        header = f.read(n).rstrip(b" ")
        header += b" " * (-(8 + len(header)) % 8)              # pad so 8 + len % 8 == 0
        g.write(struct.pack("<Q", len(header)) + header)
        shutil.copyfileobj(f, g, 64 << 20)                     # rest of file = tensor data, 64 MB chunks
