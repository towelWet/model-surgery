# _shard_name.py — HuggingFace shard filename for shard i of total.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.


def shard_name(i, total):
    """Return HuggingFace shard filename, e.g. (2,3) -> 'model-00002-of-00003.safetensors'."""
    return f"model-{i:05d}-of-{total:05d}.safetensors"
