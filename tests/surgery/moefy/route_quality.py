"""Route error: perfect logits match oracle, and oracle beats random expert choice."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import sys
from pathlib import Path
import numpy as np


def test_route_error_orders_oracle_router_random():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root: absolute pipeline.* / tests.* imports
    from pipeline.surgery.moefy._route_error import route_error
    rng = np.random.default_rng(3)
    acts = rng.normal(size=(300, 16)) * rng.exponential(size=(300, 16))  # uneven neuron importance
    wd = rng.normal(size=(8, 16))
    labels = np.repeat(np.arange(4), 4)                      # 4 experts x 4 neurons
    norms = np.stack([np.linalg.norm(acts[:, labels == e] @ wd[:, labels == e].T, axis=1) for e in range(4)], 1)
    err = route_error(acts, wd, labels, norms, 2, 4)         # logits = true norms: router == oracle
    assert np.isclose(err["router"], err["oracle"])
    assert err["oracle"] < err["random"]
    assert route_error(acts, wd, labels, norms, 4, 4)["router"] < 1e-9  # all experts = dense output
