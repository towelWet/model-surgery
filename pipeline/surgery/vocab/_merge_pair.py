# _merge_pair.py — one BPE merge entry as (left, right), for both tokenizer.json formats.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.


def merge_pair(entry):
    if isinstance(entry, str):
        left, right = entry.split(" ", 1)                     # old format: "a b" (space never in byte-level tokens)
        return left, right
    return entry[0], entry[1]                                  # new format: ["a", "b"]
