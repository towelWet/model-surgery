# ../heads/linear/_run.py — drop `drop` key-head groups (+ their value heads) in every linear-attention layer.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Ranking: captured out_proj inputs ("o.<i>", capture.py) times out_proj columns, summed per
# key group — same score as full-attention head pruning (heads/_group_scores). Config has one
# linear head count, so every linear layer drops same number (ids differ). Full-attention
# layers untouched. bf16 only (dequant/entry.py first for quantized sources).
import glob, json, os
import numpy as np
import mlx.core as mx
from pipeline.surgery.checkpoint.destination import validate_destination
from pipeline.surgery.shard_io import load_index
from pipeline.surgery._layer_prefix import decoder_layer_prefix
from pipeline.surgery.moefy._load_calib import load_calib
from pipeline.surgery.heads._group_scores import group_scores
from pipeline.surgery.heads.linear._gdn_index import gdn_index
from pipeline.surgery.heads.linear._slice_gdn import slice_gdn
from pipeline.surgery._save_pruned import save_pruned


def run_heads_linear(src, dest, drop, calib):
    dest = validate_destination(dest, sources=[src, calib])
    cfg = json.load(open(os.path.join(src, "config.json")))
    t = cfg.get("text_config", cfg)                             # VLM wrapper keeps text dims inside
    hk, hv = t["linear_num_key_heads"], t["linear_num_value_heads"]
    dk, dv = t["linear_key_head_dim"], t["linear_value_head_dim"]
    if not 0 < drop < hk:
        raise ValueError(f"drop must be in 1..{hk - 1} key-head groups")
    wmap = load_index(src)["weight_map"]
    if any(k.endswith(".scales") for k in wmap):
        raise ValueError("quantized checkpoint: run dequant/entry.py first")
    files = sorted(glob.glob(os.path.join(calib, "*.safetensors")))
    lazy = {k: v for s in sorted(set(wmap.values())) for k, v in mx.load(os.path.join(src, s)).items()}
    pre, new, report = decoder_layer_prefix(list(wmap), t["num_hidden_layers"]), {}, []
    for li, kind in enumerate(t["layer_types"]):
        if kind != "linear_attention":
            continue
        p = f"{pre}{li}.linear_attn."
        wo = np.array(lazy[p + "out_proj.weight"].astype(mx.float32))
        score = group_scores(load_calib(files, li, "o"), wo, hv, hk, dv)  # value heads grouped by key head
        keep = sorted(int(g) for g in np.argsort(-score)[: hk - drop])
        new.update(slice_gdn(lazy, p, gdn_index(keep, hk, hv, dk, dv)))
        report.append({"layer": li, "kept_groups": keep, "scores": [float(s) for s in score]})
    t.update(linear_num_key_heads=hk - drop, linear_num_value_heads=hv - drop * (hv // hk))
    save_pruned(src, dest, wmap, lazy, new, cfg)               # shards + config + tokenizer
    json.dump(report, open(dest / "heads_linear_report.json", "w"), indent=1)
    return report
