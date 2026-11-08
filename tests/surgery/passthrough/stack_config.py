"""Stacked config gets new layer count and per-layer lists re-sliced from each source."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import sys
from pathlib import Path


def test_stack_config_reslices_layer_lists():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root: absolute pipeline.* / tests.* imports
    from pipeline.surgery.passthrough.stack._config import stack_config
    a = {"num_hidden_layers": 2, "layer_types": ["full", "linear"], "hidden_size": 8}
    b = {"num_hidden_layers": 2, "layer_types": ["x", "y"], "hidden_size": 8}
    out = stack_config([a, b], [(0, 1), (1, 0), (1, 1)])
    assert out["num_hidden_layers"] == 3 and out["layer_types"] == ["linear", "x", "y"]
    assert a["num_hidden_layers"] == 2                       # source config untouched
