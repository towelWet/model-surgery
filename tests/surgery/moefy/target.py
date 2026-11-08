"""moefy target: Qwen3 -> Qwen3-MoE ("mlp"), Llama/Mistral -> Mixtral ("block_sparse_moe"), others refused."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import json
import sys
import tempfile
from pathlib import Path


def test_moe_target_and_mixtral_config():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root: absolute pipeline.* / tests.* imports
    from pipeline.surgery.moefy.moe._target import moe_target
    assert moe_target({"architectures": ["Qwen3ForCausalLM"]})[0] == "mlp"
    block, write = moe_target({"architectures": ["LlamaForCausalLM"]})
    assert block == "block_sparse_moe"
    with tempfile.TemporaryDirectory() as d:
        write(d, {"architectures": ["LlamaForCausalLM"], "intermediate_size": 8192}, 8, 2)
        c = json.load(open(Path(d) / "config.json"))
    assert c["architectures"] == ["MixtralForCausalLM"] and c["num_local_experts"] == 8
    assert c["intermediate_size"] == 1024 and c["num_experts_per_tok"] == 2   # per-expert width
    try:
        moe_target({"architectures": ["Qwen3_5ForConditionalGeneration"]})
    except ValueError as e:
        assert "pipeline/surgery/moefy/hybrid" in str(e)   # path form: fold tools rewrite it
    else:
        raise AssertionError("hybrid accepted by dense moefy")
