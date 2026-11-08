"""Write a tiny synthetic MoE checkpoint (4 experts) as raw bytes for expert-split tests.
NEVER DELETE COMMENTS — explain every non-obvious line.
Update comments with the code; do not strip them.
"""
import json
from pathlib import Path
from tests.storage.helpers.build import write_st


def moe_fixture(root, stacked=True):
    """stacked=True: MLX switch_mlp [E, ...]; False: HF per-expert mlp.experts.<e>.*. Returns src dir."""
    src = Path(root) / "src"
    tensors = [("model.embed_tokens.weight", "BF16", [2, 2], bytes(range(8))),
               ("model.layers.0.mlp.gate.weight", "BF16", [4, 2], bytes(range(100, 116))),  # router: [E, H]
               ("model.layers.0.mlp.shared_expert.up_proj.weight", "BF16", [2], bytes(4))]   # always resident
    if stacked:
        tensors.append(("model.layers.0.mlp.switch_mlp.up_proj.weight", "U32", [4, 2],
                        bytes(range(200, 232))))                     # 4 expert rows x 8 bytes, distinct per row
    else:
        tensors += [(f"model.layers.0.mlp.experts.{e}.up_proj.weight", "U32", [2], bytes([e] * 8))
                    for e in range(4)]
    write_st(src / "model.safetensors", tensors)
    (src / "config.json").write_text(json.dumps({"num_experts": 4}))
    (src / "tokenizer.json").write_text("{}")
    return src
