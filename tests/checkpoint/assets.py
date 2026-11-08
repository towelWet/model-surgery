"""Tokenizer copying preserves unseen declared assets and nested chat templates."""
import json
import tempfile
from pathlib import Path
from pipeline.surgery.shard._copy_tokenizer import copy_tokenizer
from pipeline.surgery.shard.tokenizer.discover import discover_assets


def test_assets_discovery_and_verbatim_copy():
    with tempfile.TemporaryDirectory() as tmp:
        source, output = Path(tmp) / "source", Path(tmp) / "output"
        source.mkdir()
        for name in ("tokenizer.model", "added_tokens.json", "chat_templates/tool.jinja", "custom.lex"):
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"unchanged\x00\xff")
        (source / "tokenizer_config.json").write_text(json.dumps({"vocab_file": "custom.lex"}))
        (source / "config.json").write_text("{}")
        (source / "model.safetensors").write_bytes(b"not an asset")
        copy_tokenizer(source, output)
        assert not (output / "config.json").exists()
        assert not (output / "model.safetensors").exists()
        for relative in discover_assets(source):
            assert (output / relative).read_bytes() == (source / relative).read_bytes()
        formats = Path(tmp) / "formats"
        formats.mkdir()
        (source / "unseen.asset").write_bytes(b"new format")
        (formats / "custom.toml").write_text('patterns = ["*.asset"]')
        assert Path("unseen.asset") in discover_assets(source, formats)
