"""Forged manifest asset path rejected semantically even when checksum is recomputed."""
import hashlib
import tempfile
from pathlib import Path
from pipeline.storage.reassemble import reassemble_checkpoint
from pipeline.storage.split import split_checkpoint
from tests.storage.helpers.build import write_st


def test_forged_asset_path_rejected_with_valid_checksum():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src"
        src.mkdir()
        write_st(src / "w.safetensors", [("a", "F32", [1], b"\x01" * 4)])
        (src / "config.json").write_text("{}")
        store = Path(tmp) / "store-Tau"
        split_checkpoint(src, store)
        manifest = store / "manifest.toml"
        text = manifest.read_text()
        manifest.write_text(text.replace('path = "config.json"', 'path = "../outside"', 1))
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        (store / "complete.toml").write_text(f'manifest_sha256 = "{digest}"\n')
        dest = Path(tmp) / "out-Tau"
        try:
            reassemble_checkpoint(store, dest)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted forged asset path")
        assert not dest.exists() and not (Path(tmp) / "outside").exists()
