"""AWQ fold keeps float outputs exact; chosen scale never worse than plain RTN (alpha = 0)."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import sys
from pathlib import Path
import numpy as np


def test_awq_scale_and_fold():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root: absolute pipeline.* / tests.* imports
    from pipeline.surgery.wquant.awq._scale import awq_scale
    from pipeline.surgery.wquant._rtn_dequant import rtn_dequant
    rng = np.random.default_rng(1)
    w = rng.normal(size=(32, 128)).astype(np.float32)
    x = rng.normal(size=(512, 128)).astype(np.float32) * np.where(np.arange(128) < 4, 30.0, 1.0)  # 4 salient channels
    s = awq_scale([w], x, 4, 64)
    err = lambda wq: np.square(x @ w.T - x @ wq.T).mean()
    plain = err(rtn_dequant(w, 4, 64))
    scaled = err(rtn_dequant(w * s, 4, 64) / s)
    assert scaled <= plain + 1e-9 and np.allclose((x / s) @ (w * s).T, x @ w.T, rtol=1e-4, atol=1e-3)
