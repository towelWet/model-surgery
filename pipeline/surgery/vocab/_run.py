# _run.py — vocabulary pruning: keep tokens a corpus needs, drop embedding/head rows.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Text that only uses kept tokens tokenizes to same pieces (ids renumbered). Other text still
# encodes (byte alphabet kept) but splits into more, shorter tokens. Quantized rows slice fine:
# packing runs along hidden dim, rows are axis 0 of weight, scales and biases alike.
import json
import os
import mlx.core as mx
from pipeline.surgery.checkpoint.destination import validate_destination
from pipeline.surgery.shard_io import copy_tokenizer, load_index
from pipeline.surgery.vocab._corpus_ids import corpus_ids
from pipeline.surgery.vocab._bpe_closure import bpe_closure
from pipeline.surgery.vocab._trim_tokenizer import trim_tokenizer
from pipeline.surgery.vocab._remap_token_ids import remap_token_ids
from pipeline.surgery.vocab._write_slow_files import write_slow_files

ROW_PREFIXES = ("model.embed_tokens.", "lm_head.")             # tensors indexed by token id


def run_vocab(src, dest, corpus_files):
    dest = validate_destination(dest, sources=[src])
    tj = json.load(open(os.path.join(src, "tokenizer.json")))
    kept = bpe_closure(tj, corpus_ids(os.path.join(src, "tokenizer.json"), corpus_files))
    remap = {old: new for new, old in enumerate(kept)}
    rows = mx.array(kept)
    os.makedirs(dest)
    for shard in sorted(set(load_index(src)["weight_map"].values())):
        w = mx.load(os.path.join(src, shard))
        out = {k: mx.take(v, rows, axis=0) if k.startswith(ROW_PREFIXES) else v for k, v in w.items()}
        mx.eval(list(out.values()))
        mx.save_safetensors(str(dest / shard), out, metadata={"format": "mlx"})
    copy_tokenizer(src, dest)
    new_tj = trim_tokenizer(tj, remap)
    json.dump(new_tj, open(dest / "tokenizer.json", "w"), ensure_ascii=False)
    write_slow_files(dest, new_tj)
    for name in ("config.json", "generation_config.json", "tokenizer_config.json"):
        if os.path.exists(os.path.join(src, name)):
            obj = remap_token_ids(json.load(open(os.path.join(src, name))), remap)
            if name == "config.json":
                obj["vocab_size"] = len(kept)
            json.dump(obj, open(dest / name, "w"), indent=2, ensure_ascii=False)
    if os.path.exists(os.path.join(src, "model.safetensors.index.json")):
        json.dump(load_index(src), open(dest / "model.safetensors.index.json", "w"), indent=2)
    return len(kept), len(tj["model"]["vocab"]) + len(tj["added_tokens"])
