# stack/_core.py — passthrough merge: copy chosen layers of several models into one checkpoint.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Tensors are copied untouched (quantized bytes included). Embeddings, final norm and
# lm_head come from source 0. One output shard per 8 stacked layers bounds peak memory.
import json, os
import mlx.core as mx
from pipeline.surgery.checkpoint.destination import validate_destination
from pipeline.surgery.shard_io import copy_tokenizer, load_index, write_index
from pipeline.surgery._layer_prefix import decoder_layer_prefix
from pipeline.surgery.passthrough._parse_slices import parse_slices
from pipeline.surgery.passthrough._check_compatible import check_compatible
from pipeline.surgery.passthrough.stack._config import stack_config


def stack(sources, dest, spec):
    dest = validate_destination(dest, sources=sources)
    cfgs = [json.load(open(os.path.join(s, "config.json"))) for s in sources]
    check_compatible(cfgs)
    slices = parse_slices(spec, [c["num_hidden_layers"] for c in cfgs])
    maps, lazy = [load_index(s)["weight_map"] for s in sources], []
    for s, m in zip(sources, maps):
        w = {}
        for shard in sorted(set(m.values())):
            w.update(mx.load(os.path.join(s, shard)))        # lazy: only copied keys are read
        lazy.append(w)
    pre = [decoder_layer_prefix(list(m), c["num_hidden_layers"]) for m, c in zip(maps, cfgs)]
    os.makedirs(dest)
    groups = [{k: v for k, v in lazy[0].items() if not k.startswith(pre[0])}]  # non-layer tensors
    for j, (s, l) in enumerate(slices):
        if j % 8 == 0:
            groups.append({})
        old = f"{pre[s]}{l}."
        groups[-1].update({f"{pre[0]}{j}.{k[len(old):]}": v for k, v in lazy[s].items() if k.startswith(old)})
    wmap = {}
    for i, g in enumerate(groups):
        name = f"model-{i + 1:05d}-of-{len(groups):05d}.safetensors"
        mx.eval(*g.values())
        mx.save_safetensors(str(dest / name), g, metadata={"format": "mlx"})
        wmap.update({k: name for k in g})
    write_index(dest, {}, wmap)
    json.dump(stack_config(cfgs, slices), open(dest / "config.json", "w"), indent=2)
    copy_tokenizer(sources[0], dest)
    return slices
