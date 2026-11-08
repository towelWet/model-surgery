"""Destination symlink traversal rejected before any destination write."""
import json
import tempfile
from pathlib import Path
from pipeline.surgery.shard._copy_tokenizer import copy_tokenizer


def test_destination_symlink_rejected_without_writes():
    with tempfile.TemporaryDirectory() as tmp:
        source, output = Path(tmp) / "source", Path(tmp) / "output"
        outside = Path(tmp) / "outside"
        (source / "sub").mkdir(parents=True)
        output.mkdir()
        outside.mkdir()
        (source / "sub" / "evil.lex").write_bytes(b"payload")
        (source / "tokenizer_config.json").write_text(json.dumps({"vocab_files": ["sub/evil.lex"]}))
        (output / "sub").symlink_to(outside)                # traversal via dest symlink
        try:
            copy_tokenizer(source, output)
        except ValueError:
            pass
        else:
            raise AssertionError("Accepted destination symlink traversal")
        assert not (outside / "evil.lex").exists()          # nothing escaped
        assert (source / "sub" / "evil.lex").read_bytes() == b"payload"
