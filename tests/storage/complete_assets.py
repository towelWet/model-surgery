"""Round-trip every regular file, including multimodal assets and unindexed shard-like files."""
import tempfile
import json
from pathlib import Path
from pipeline.storage.split import split_checkpoint
from pipeline.storage.reassemble import reassemble_checkpoint
from tests.storage.helpers.build import write_st


def test_entire_checkpoint_asset_tree_roundtrips():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "source"
        src.mkdir()
        write_st(src / "weights.safetensors", [("w", "F32", [1], b"\x00" * 4)])
        assets = {"config.json": b"{}", "processor_config.json": b"{}",
                  "vision/preprocess.bin": b"vision", "audio/vocoder.dat": b"audio",
                  "custom/model.py": b"# model source\n", ".hidden": b"hidden",
                  "aux/custom.safetensors": b"unindexed auxiliary payload",
                  "model.safetensors.index.json": json.dumps({"weight_map": {"w": "weights.safetensors"}}).encode()}
        for rel, content in assets.items():
            path = src / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        plan = split_checkpoint(src, Path(tmp) / "store")
        assert {a["path"] for a in plan["assets"]} == set(assets)
        output = Path(tmp) / "joined"
        reassemble_checkpoint(Path(tmp) / "store", output)
        for path in src.rglob("*"):
            if path.is_file():
                assert (output / path.relative_to(src)).read_bytes() == path.read_bytes()
