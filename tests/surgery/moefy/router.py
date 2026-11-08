"""Ridge router fit recovers a known linear map from MLP input to expert importance."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import sys
from pathlib import Path
import numpy as np


def test_fit_router_recovers_linear_map():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root: absolute pipeline.* / tests.* imports
    from pipeline.surgery.moefy._fit_router import fit_router
    rng = np.random.default_rng(2)
    x = rng.normal(size=(2000, 16))
    true = rng.normal(size=(16, 4))
    r = fit_router(x, x @ true, ridge=1e-6)                  # near-zero ridge: exact recovery expected
    assert np.allclose(r, true, atol=1e-3)
