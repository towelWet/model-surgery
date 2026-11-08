"""GPTQ: packs like MLX, equals RTN when H = I, beats RTN on output error with correlated inputs."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import sys
from pathlib import Path
import numpy as np
import mlx.core as mx


def test_gptq_matches_rtn_on_identity_and_wins_on_real_cov():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root: absolute pipeline.* / tests.* imports
    from pipeline.surgery.wquant._gptq_matrix import gptq_matrix
    from pipeline.surgery.wquant._pack_affine import pack_affine
    rng = np.random.default_rng(0)
    w = rng.normal(size=(16, 128)).astype(np.float32)
    codes, s, b = gptq_matrix(w, np.eye(128), 4, 64, damp=0.0)
    deq = np.array(mx.dequantize(mx.array(pack_affine(codes, 4)), mx.array(s), mx.array(b), group_size=64, bits=4))
    q, s0, b0 = mx.quantize(mx.array(w), group_size=64, bits=4)
    rtn = np.array(mx.dequantize(q, s0, b0, group_size=64, bits=4))
    assert np.abs(deq - rtn).mean() < 0.02                   # same codes; only bf16-rounded scales differ (step ~0.3)
    mix = rng.normal(size=(128, 128)) * 0.3 + np.eye(128)      # correlated input channels
    x = (rng.normal(size=(2048, 128)) @ mix).astype(np.float32)
    codes, s, b = gptq_matrix(w, 2 * x.T @ x / len(x), 4, 64)
    deq = np.array(mx.dequantize(mx.array(pack_affine(codes, 4)), mx.array(s), mx.array(b), group_size=64, bits=4))
    err = lambda wq: np.square(x @ (w - wq).T).mean()
    assert err(deq) < 0.8 * err(rtn)                          # error feedback pays off on real covariance
