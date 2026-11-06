"""split->join restores byte-identical shards + assets incl __metadata__ and index."""
import json
import tempfile
from pathlib import Path
from pipeline.storage.split import split_checkpoint
from pipeline.storage.reassemble import reassemble_checkpoint
from tests.storage.helpers.build import write_st


def test_split_join_byte_identical_checkpoint():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src"
        src.mkdir()
        write_st(src / "a.safetensors", [
            ("w.bf16", "BF16", [2, 2], b"\x01\x02\x03\x04\x05\x06\x07\x08"),
            ("w.u32", "U32", [2], b"\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10")],
            metadata={"format": "pt"})
        write_st(src / "nested" / "b.safetensors", [("w.f32", "F32", [1], b"\xaa\xbb\xcc\xdd")])
        (src / "config.json").write_text('{"model_type": "fixture"}')
        (src / "tokenizer_config.json").write_text('{"vocab_file": "vocab.model"}')
        (src / "vocab.model").write_bytes(b"vocab")
        (src / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
            "w.bf16": "a.safetensors", "w.u32": "a.safetensors",
            "w.f32": "nested/b.safetensors"}, "metadata": {}}))
        store = Path(tmp) / "store-Tau"
        split_checkpoint(src, store)
        out = Path(tmp) / "out-Tau"
        reassemble_checkpoint(store, out)
        for rel in ("a.safetensors", "nested/b.safetensors", "config.json",
                    "vocab.model", "tokenizer_config.json", "model.safetensors.index.json"):
            assert (out / rel).read_bytes() == (src / rel).read_bytes(), rel
