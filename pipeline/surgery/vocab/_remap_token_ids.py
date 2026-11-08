# _remap_token_ids.py — rewrite *_token_id fields (int or list) and added_tokens_decoder keys.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Used on config.json, generation_config.json and tokenizer_config.json. Ids not kept raise:
# a special id pointing at a dropped row would silently read wrong embedding.


def remap_token_ids(obj, remap):
    out = dict(obj)
    for k, v in obj.items():
        if k.endswith("_token_id") and v is not None:
            out[k] = [remap[i] for i in v] if isinstance(v, list) else remap[v]
    if "added_tokens_decoder" in obj:
        out["added_tokens_decoder"] = {str(remap[int(i)]): t for i, t in obj["added_tokens_decoder"].items()}
    return out
