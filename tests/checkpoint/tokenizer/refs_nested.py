"""Nested objects still recurse; null and non-string refs ignored; bad plural refs rejected."""
from pathlib import Path
from pipeline.surgery.shard.tokenizer.references import referenced_assets


def test_nested_refs_and_unsafe_plural_rejected():
    metadata = {"outer": {"inner": {"merges_file": "m.txt", "vocab_file": None}}}
    assert referenced_assets(metadata) == {Path("m.txt")}
    for bad in ({"vocab_files": ["../up.lex"]}, {"aux_files": {"k": "/abs.lex"}}):
        try:
            referenced_assets(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted unsafe plural ref: {bad}")
