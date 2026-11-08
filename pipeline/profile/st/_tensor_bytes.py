# st/_tensor_bytes.py — read one tensor's raw bytes by seek, keeping RAM flat on 5 GB shards.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.


def tensor_bytes(loc):
    """Raw little-endian bytes of one tensor; loc = (shard_path, entry, data_start) from index()."""
    path, entry, start = loc
    a, b = entry["data_offsets"]                             # [start, end) inside data blob
    with open(path, "rb") as f:
        f.seek(start + a)
        return f.read(b - a)
