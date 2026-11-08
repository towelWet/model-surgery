"""Wanda keeps small weights on loud input channels that magnitude pruning would drop."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import sys
from pathlib import Path
import numpy as np
import mlx.core as mx


def test_wanda_uses_activation_norms():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root: absolute pipeline.* / tests.* imports
    from pipeline.surgery.sparse.prune._weight import prune_weight
    w = mx.array([[0.1, 1.0]]).astype(mx.bfloat16)          # channel 0 small weight, channel 1 big
    mag, zero = prune_weight(w, 0.5)
    assert np.array(mag.astype(mx.float32)).tolist()[0][0] == 0.0 and zero == 0.5
    wanda, _ = prune_weight(w, 0.5, norms=np.array([100.0, 1.0]))  # channel 0 input is 100x louder
    out = np.array(wanda.astype(mx.float32))[0]
    assert out[1] == 0.0 and abs(out[0] - 0.1) < 1e-2        # kept value is original bf16 weight
