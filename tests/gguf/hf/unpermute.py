"""Llama q/k: unpermute_qk inverts llama.cpp LlamaModel.permute exactly (q heads and GQA k heads)."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import sys
from pathlib import Path
import numpy as np


def test_unpermute_inverts_llamacpp_permute():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root: absolute pipeline.* / tests.* imports
    import mlx.core as mx                                       # lazy: profile tests assert mlx absent, run later
    from pipeline.gguf.hf.unpermute_qk import unpermute_qk
    rng = np.random.default_rng(3)
    for heads, d, cols in ((4, 8, 5), (2, 8, 5)):              # q-like and kv-like shapes
        w = rng.normal(size=(heads * d, cols)).astype(np.float32)
        # copy of convert_hf_to_gguf.py LlamaModel.permute: the transform being inverted
        p = w.reshape(heads, 2, d // 2, cols).swapaxes(1, 2).reshape(w.shape)
        assert not np.array_equal(p, w)                         # permutation really moves rows
        assert np.array_equal(np.array(unpermute_qk(mx.array(p), heads)), w)
