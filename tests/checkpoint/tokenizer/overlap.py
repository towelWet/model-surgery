"""Identical or nested source/destination pairs rejected before any write."""
import json
import tempfile
from pathlib import Path
from pipeline.surgery.shard._copy_tokenizer import copy_tokenizer


def test_overlapping_dirs_rejected_without_writes():
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "outer" / "source"
        source.mkdir(parents=True)
        (source / "tokenizer_config.json").write_text(json.dumps({"vocab_file": "v.lex"}))
        (source / "v.lex").write_bytes(b"vocab")
        nested, parent = source / "nested", Path(tmp) / "outer"
        for dest in (source, nested, parent):
            try:
                copy_tokenizer(source, dest)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Accepted overlapping destination: {dest}")
        assert not nested.exists()                          # no partial destination writes
        assert not (parent / "v.lex").exists()
        assert (source / "v.lex").read_bytes() == b"vocab"  # source untouched
