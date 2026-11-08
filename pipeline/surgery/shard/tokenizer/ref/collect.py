"""Collect path references from one metadata key ('_file' single, '_files' list/dict)."""
from .add import add_ref


def collect_refs(result, key, item):
    if key.endswith("_file"):
        add_ref(result, key, item)                             # single path: "vocab_file"
    elif key.endswith("_files"):
        entries = item.values() if isinstance(item, dict) else (item if isinstance(item, list) else ())
        for entry in entries:
            add_ref(result, key, entry)                        # list/dict of paths: "vocab_files"
