# is_quantized_key.py — True when key is a .weight with sibling scales/biases in the index.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.


def is_quantized_key(key, wmap):
    """True when key is a .weight with sibling scales/biases in the index."""
    if not key.endswith(".weight"):
        return False
    prefix = key[: -len(".weight")]
    return prefix + ".scales" in wmap and prefix + ".biases" in wmap
