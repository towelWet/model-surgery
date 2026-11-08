"""Validate and record one local tokenizer asset reference."""
from pathlib import Path


def add_ref(result, key, item):
    if not isinstance(item, str):
        return                                                 # null/non-string refs ignored
    path = Path(item)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Non-local tokenizer asset reference: {key}")
    result.add(path)
