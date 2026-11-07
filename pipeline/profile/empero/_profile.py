# _profile.py — CPU-only Empero dense-FFN profile for post-hoc MoE.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "surgery"))         # _index_or_single, _whittle …
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "surgery", "mlp"))  # _hw lives in surgery/mlp/
from _index_or_single import synthesize_index  # not shard_io: that imports mlx
from _layer_prefix import decoder_layer_prefix
from _hw import mlp_hw
from _safetensors_header import safetensors_header
from _whittle import expert_width


def empero_profile(model_dir, experts=64):
    """Measure Empero MLP dims from config + safetensors header. No GPU, no :8765."""
    cfg = json.loads(open(os.path.join(model_dir, "config.json")).read())
    tc = cfg.get("text_config", cfg)
    hidden, intermediate = mlp_hw(tc)
    width = expert_width(intermediate, hidden, experts)
    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    single_file = not os.path.exists(idx_path)
    keys = list((json.loads(open(idx_path).read()) if not single_file
                 else synthesize_index(model_dir))["weight_map"])
    prefix = decoder_layer_prefix(keys, tc.get("num_hidden_layers"))  # VLM text stack by layer count
    header = safetensors_header(os.path.join(model_dir, "model.safetensors")) if single_file else {}
    return {
        "model_dir": model_dir,
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_hidden_layers": tc.get("num_hidden_layers"),
        "n_tensors": len(keys),
        "decoder_prefix": prefix,
        "gate_proj_shape": header.get(prefix + "0.mlp.gate_proj.weight", {}).get("shape"),
        "layer_type_counts": dict(Counter(tc.get("layer_types", []))),
        "mtp": any(k.startswith("mtp.") for k in keys),
        "single_file": single_file,
        "whittle": {"experts": experts, "expert_width": width, "shared": hidden,
                    "identity": f"{experts}*{width}+{hidden}={intermediate}",
                    "active_topk4": 4 * width + hidden},
    }
