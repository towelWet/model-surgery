"""Unedited read+write reproduces GGUF bytes exactly; existing destinations are refused."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import tempfile
from pathlib import Path
from pipeline.gguf.format.read.gguf import read_gguf
from pipeline.gguf.format.write.gguf import write_gguf
from tests.gguf.helpers.moe_fixture import moe_fixture


def test_roundtrip_is_byte_identical_and_never_overwrites():
    with tempfile.TemporaryDirectory() as tmp:
        src = moe_fixture(str(Path(tmp) / "src.gguf"))
        out = str(Path(tmp) / "out.gguf")
        write_gguf(src, out, read_gguf(src))
        assert Path(src).read_bytes() == Path(out).read_bytes()
        try:
            write_gguf(src, out, read_gguf(src))
        except FileExistsError:
            return
        raise AssertionError("overwrote existing destination")
