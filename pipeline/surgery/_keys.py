# _keys.py — layer key remapping and shard naming utilities.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import re

from _layer_prefix import DECODER_PREFIXES   # Tau + Empero nested VLM prefixes
from _shard_name import shard_name           # re-export: many callers import it from _keys


def remap_key(key, remove_set):
    """Remap layer index in a weight key; return None if layer is in remove_set."""
    prefix = next((p for p in DECODER_PREFIXES if key.startswith(p)), None)
    if prefix is None:
        return key                           # non-decoder key (embed, MTP, vision) passes through
    m = re.match(r"^(\d+)(\..+)$", key[len(prefix):])
    if not m:
        return key
    idx = int(m.group(1))
    if idx in remove_set:
        return None                          # this layer is being removed
    gap = sum(1 for r in remove_set if r < idx)  # layers removed below this one
    return prefix + str(idx - gap) + m.group(2)
