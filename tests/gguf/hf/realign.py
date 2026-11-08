"""Realign: MLX-written safetensors gets 8-byte data start; tensors load back bit-identical."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import sys
import tempfile
from pathlib import Path


def test_realign_pads_header_and_keeps_tensors():
    import mlx.core as mx                                       # lazy: profile tests assert mlx absent, run later
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root: absolute pipeline.* / tests.* imports
    from pipeline.gguf.hf.is_aligned import is_aligned
    from pipeline.gguf.hf.realign_file import realign_file
    d = Path(tempfile.mkdtemp())                                # TMPDIR decides location
    t = {"a": mx.arange(6, dtype=mx.bfloat16).reshape(2, 3), "bb": mx.ones((5,), dtype=mx.float32)}
    for pad in range(8):                                        # some name length gives unaligned header
        src = d / f"s{pad}.safetensors"
        mx.save_safetensors(str(src), {**t, "x" * (pad + 1): mx.zeros((1,))})
        if not is_aligned(src):
            break
    assert not is_aligned(src)
    realign_file(src, d / "out.safetensors")
    assert is_aligned(d / "out.safetensors")
    back = mx.load(str(d / "out.safetensors"))
    assert all(mx.array_equal(back[k], v).item() for k, v in t.items())
