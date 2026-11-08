"""Combined-MoE router sends each expert's own prompts to that expert; layer stacks whole MLPs."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import tempfile
from pathlib import Path
import numpy as np
import mlx.core as mx


def test_router_rows_and_stack():
    from pipeline.surgery.moe.combine._router_rows import router_rows
    from pipeline.surgery.moe.combine._layer import combine_layer
    rng = np.random.default_rng(0)
    shared = rng.normal(size=64) * 5                           # direction every prompt shares
    tmp = Path(tempfile.mkdtemp(dir=tempfile.gettempdir()))
    sets = []
    for e in range(2):
        x = shared + rng.normal(size=(40, 64)) + 3 * np.eye(64)[e]  # expert e prompts lean on axis e
        f = tmp / f"e{e}.safetensors"
        mx.save_safetensors(str(f), {"mlp.0": mx.array(x.astype(np.float32))})
        sets.append([str(f)])
    rows = router_rows(sets, 0)
    for e in range(2):
        x = np.array(mx.load(sets[e][0])["mlp.0"])
        assert ((x @ rows.T).argmax(1) == e).mean() > 0.95  # own prompts route home
    w = [{f"model.layers.0.mlp.{n}.weight": mx.full((64, 64), float(i + 1)) for n in ("gate_proj", "up_proj", "down_proj")}
         for i in range(2)]
    out = combine_layer(0, w, rows.astype(np.float32), 64, 8)
    deq = mx.dequantize(out["model.layers.0.mlp.switch_mlp.up_proj.weight"], out["model.layers.0.mlp.switch_mlp.up_proj.scales"],
                        out["model.layers.0.mlp.switch_mlp.up_proj.biases"], 64, 8)
    assert deq.shape == (2, 64, 64) and abs(deq[1].mean().item() - 2.0) < 0.02  # expert 1 = second fine-tune
