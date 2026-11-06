# _moefy.py — dense FFN → post-hoc MoE partition; no CLI.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import json
import os

import sys as _sys
_sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # surgery/ parent

from _emit import make_emitter                           # buffered weight writer
from _gate import emit_gates                             # zero-init router gates
from _partition_loop import partition_weights             # MLP partition loop
from _save_moefy import save_moefy                             # checkpoint writer
from _moe_config import write_moe_config                 # dest config.json
from mlp._hw import mlp_hw                               # hidden/intermediate dims
from _whittle import expert_width                         # expert shard width
from shard_io import copy_tokenizer, load_index           # safetensors I/O
from checkpoint.destination import validate_destination   # shared dest guard
from _expert_counts import check_counts                    # E/K sanity


def moefy(src, dest, experts, topk):
    """Partition dense FFN into MoE: shared + E experts per layer."""
    validate_destination(dest, sources=[src])               # Tau name + overlap before any writes
    E, K = experts, topk
    check_counts(E, K)
    with open(os.path.join(src, "config.json")) as f:
        cfg = json.load(f)
    tc = cfg.get("text_config", cfg)
    H, W = mlp_hw(tc)                                   # Tau 5120/17408 or Empero 4096/12288
    S = expert_width(W, H, E)
    q = cfg.get("quantization") or {}
    bits = q.get("bits") if q else None                  # None = dense bf16 (Empero)
    gsize = q.get("group_size", 64)
    if E * S + H != W:
        raise ValueError(f"partition mismatch: {E}*{S}+{H} != {W}")
    print(f"MoE partition: {W} = {E}×{S} + {H} shared  (top-{K} → {K*S+H} active/token)")

    idx = load_index(src)
    wmap = idx["weight_map"]
    n_layers = tc["num_hidden_layers"]
    os.makedirs(dest)                                    # all validation passed; create output
    copy_tokenizer(src, dest)
    write_moe_config(dest, cfg, tc, E, K, S, H)

    out = {}
    emit = make_emitter(out)                             # buffered weight emitter
    partition_weights(src, wmap, emit, E, S, H, bits, gsize)
    emit_gates(wmap, emit, n_layers, E, H)
    save_moefy(dest, out, wmap, idx.get("metadata", {}), K, S, H, W)
