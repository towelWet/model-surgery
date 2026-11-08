# _run.py — attention-head pruning: per layer drop the `drop` least important KV groups.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Ranking uses captured per-head outputs ("o.<i>" from profile/moefy/capture.py). Runtime
# config has one head count for all layers, so every layer drops same number (not same ids). bf16 only.
# Hybrids (Qwen3.5/3.6): dims from text_config, only "full_attention" layers touched (linear
# layers: heads_linear/), q_proj rows per head doubled when it packs [query | output gate].
import glob, json, os
import numpy as np
import mlx.core as mx
from pipeline.surgery.checkpoint.destination import validate_destination
from pipeline.surgery.shard_io import load_index
from pipeline.surgery._layer_prefix import decoder_layer_prefix
from pipeline.surgery.moefy._load_calib import load_calib
from pipeline.surgery.heads._group_scores import group_scores
from pipeline.surgery.heads._slice_attention import slice_attention
from pipeline.surgery._save_pruned import save_pruned


def run_heads(src, dest, drop, calib):
    dest = validate_destination(dest, sources=[src, calib])
    cfg = json.load(open(os.path.join(src, "config.json")))
    t = cfg.get("text_config", cfg)                             # VLM wrapper keeps text dims inside
    n_q, n_kv, d = t["num_attention_heads"], t["num_key_value_heads"], t["head_dim"]
    if not 0 < drop < n_kv:
        raise ValueError(f"drop must be in 1..{n_kv - 1} KV groups")
    wmap = load_index(src)["weight_map"]
    if any(k.endswith(".scales") for k in wmap):
        raise ValueError("quantized checkpoint: prune bf16 source first (dequant/entry.py), then quantize")
    files = sorted(glob.glob(os.path.join(calib, "*.safetensors")))
    lazy = {k: v for s in sorted(set(wmap.values())) for k, v in mx.load(os.path.join(src, s)).items()}
    pre, new, report = decoder_layer_prefix(list(wmap), t["num_hidden_layers"]), {}, []
    kinds = t.get("layer_types") or ["full_attention"] * t["num_hidden_layers"]
    for li in (i for i, k in enumerate(kinds) if k == "full_attention"):
        p = f"{pre}{li}."
        q_rows = lazy[p + "self_attn.q_proj.weight"].shape[0] // n_q   # d, or 2d when gated
        wo = np.array(lazy[p + "self_attn.o_proj.weight"].astype(mx.float32))
        score = group_scores(load_calib(files, li, "o"), wo, n_q, n_kv, d)
        keep = sorted(int(g) for g in np.argsort(-score)[: n_kv - drop])  # ascending: head order kept
        new.update(slice_attention(lazy, p, keep, n_q // n_kv, d, q_rows))
        report.append({"layer": li, "kept_groups": keep, "scores": [float(s) for s in score]})
    t.update(num_attention_heads=n_q - drop * (n_q // n_kv), num_key_value_heads=n_kv - drop)
    save_pruned(src, dest, wmap, lazy, new, cfg)               # shards + config + index + tokenizer
    json.dump(report, open(dest / "heads_report.json", "w"), indent=1)
    return report
