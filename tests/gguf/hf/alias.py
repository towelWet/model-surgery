"""GGUF->HF name: reference names win over other-arch aliases; Llama style is fallback."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from types import SimpleNamespace
from pipeline.gguf.hf.name import hf_name


def test_alias_choice():
    tm = SimpleNamespace(mapping={"model.layers.0.attention.wq": (0, "blk.0.attn_q"),   # other arch, same prefix
                                  "model.layers.0.self_attn.q_proj": (0, "blk.0.attn_q"),
                                  "transformer.h.0.attn.q": (0, "blk.0.attn_q"),
                                  "blk.0.attn_q": (0, "blk.0.attn_q")})
    assert hf_name(tm, "blk.0.attn_q.weight", {"model.layers.0.attention.wq.weight"}) == "model.layers.0.attention.wq.weight"
    assert hf_name(tm, "blk.0.attn_q.weight") == "model.layers.0.self_attn.q_proj.weight"
