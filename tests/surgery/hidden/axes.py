"""Hidden-size axis: readers slice inputs, writers slice outputs, head norms untouched."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import sys
from pathlib import Path


def test_hidden_axis_roles():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root: absolute pipeline.* / tests.* imports
    from pipeline.surgery.hidden._axis import hidden_axis
    h = 1024
    assert hidden_axis("model.layers.0.self_attn.k_proj.weight", (1024, 1024), h) == 1  # square: by role
    assert hidden_axis("model.layers.0.self_attn.o_proj.weight", (1024, 2048), h) == 0
    assert hidden_axis("model.layers.0.mlp.down_proj.weight", (1024, 3072), h) == 0
    assert hidden_axis("model.embed_tokens.weight", (151936, 1024), h) == 1
    assert hidden_axis("model.layers.0.input_layernorm.weight", (1024,), h) == 0
    assert hidden_axis("model.layers.0.self_attn.q_norm.weight", (128,), h) is None
