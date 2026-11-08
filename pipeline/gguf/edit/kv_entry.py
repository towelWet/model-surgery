# kv_entry.py — find one metadata entry by key (returns None when absent).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.


def kv_entry(model, key):
    return next((e for e in model["kv"] if e["key"] == key), None)
