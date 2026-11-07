# _layer_prefix.py — discover indexed block namespaces from checkpoint keys.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.

import re


def decoder_layer_prefix(keys, layers=None):
    """One indexed namespace is discoverable; several need config's decoder layer count to disambiguate."""
    indices = {}                                                  # namespace -> distinct block indices
    for key in keys:
        match = re.match(r"^(.*?\.)(\d+)\..+$", key)
        if match:
            indices.setdefault(match.group(1), set()).add(int(match.group(2)))
    prefixes = set(indices)
    if len(prefixes) > 1 and type(layers) is int:                 # VLM: text + vision (+ mtp) stacks
        prefixes = {p for p in prefixes if indices[p] == set(range(layers))}  # exact 0..N-1 coverage only
    if len(prefixes) != 1:                                        # equal-depth stacks stay refused
        raise ValueError(f"expected one unambiguous block namespace; found {sorted(indices)}")
    return next(iter(prefixes))
