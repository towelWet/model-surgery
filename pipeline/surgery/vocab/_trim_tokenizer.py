# _trim_tokenizer.py — tokenizer.json with only kept tokens, ids renumbered 0..n-1 in old order.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import copy
from pipeline.surgery.vocab._merge_pair import merge_pair


def trim_tokenizer(tj, remap):
    """remap: old id -> new id. Merges survive only if both parts and result are kept."""
    out = copy.deepcopy(tj)
    vocab = {t: remap[i] for t, i in tj["model"]["vocab"].items() if i in remap}
    out["model"]["vocab"] = vocab
    out["model"]["merges"] = [m for m in tj["model"]["merges"]
                              if all(p in vocab for p in (*merge_pair(m), "".join(merge_pair(m))))]
    for tok in out["added_tokens"]:
        tok["id"] = remap[tok["id"]]                           # added tokens always kept by closure
    return out
