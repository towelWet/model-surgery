"""Escaping asset references and symlinks fail before any destination write."""
import json
import tempfile
from pathlib import Path
from pipeline.surgery.shard._copy_tokenizer import copy_tokenizer


def test_assets_reject_escape_before_copy():
    with tempfile.TemporaryDirectory() as tmp:
        source, output = Path(tmp) / "source", Path(tmp) / "output"
        source.mkdir()
        outside = Path(tmp) / "private"
        outside.write_text("private")
        config = source / "tokenizer_config.json"
        for ref in ("../private", str(outside), "missing.lex"):
            config.write_text(json.dumps({"vocab_file": ref}))
            try:
                copy_tokenizer(source, output)
            except (ValueError, FileNotFoundError):
                pass
            else:
                raise AssertionError(f"Accepted unsafe reference: {ref}")
            assert not output.exists()
        config.write_text("{}")
        (source / "tokenizer.model").symlink_to(outside)
        try:
            copy_tokenizer(source, output)
        except ValueError:
            pass
        else:
            raise AssertionError("Accepted escaping symlink")
        assert not output.exists()
