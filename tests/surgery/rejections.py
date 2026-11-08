"""Invalid neuron specs and ambiguous quantization fail before any output exists."""
import sys
import tempfile
from pathlib import Path
import mlx.core as mx
from tests.checkpoint.helpers.mlx import write_mlx_checkpoint


def test_neuron_rejects_unsupported_or_invalid_inputs_before_writing():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root: absolute pipeline.* / tests.* imports
    from pipeline.surgery.neuron._apply_edit import apply_neuron_edit
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        key = "decoder.blocks.0.mlp.down_proj.weight"
        q, s, b = mx.quantize(mx.ones((2, 32)), bits=4, group_size=32)
        cfg = {"quantization": {"bits": 4, "group_size": 32}}
        src = write_mlx_checkpoint(root / "source", {key: q, key.replace(".weight", ".scales"): s,
                                   key.replace(".weight", ".biases"): b}, cfg)
        spec = {"layer": 0, "proj": "down_proj.weight", "columns": [0], "direction": [1, -1], "alpha": 1}
        cases = [({**spec, "columns": [0, 0]}, cfg), ({**spec, "columns": [-1]}, cfg),
                 ({**spec, "direction": [1]}, cfg), (spec, {}),
                 (spec, {"quantization": {"bits": 4, "group_size": 32, "mode": "mxfp4"}})]
        for i, (edit, config) in enumerate(cases):
            dest = root / f"invalid-{i}"
            try:
                apply_neuron_edit(src, dest, edit, config)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid edit accepted")
            assert not dest.exists()
