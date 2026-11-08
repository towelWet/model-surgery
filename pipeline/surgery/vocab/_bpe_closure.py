# _bpe_closure.py — grow kept token set so BPE stays valid: alphabet + added + merge parents.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Byte-level BPE can spell any text from its single-character alphabet, so keeping all
# one-character tokens keeps tokenizer complete. Every kept token formed by merge (a, b)
# also keeps a and b, so BPE reaches it by same steps it used before.
from pipeline.surgery.vocab._merge_pair import merge_pair


def bpe_closure(tj, used_ids):
    vocab = tj["model"]["vocab"]
    by_id = {i: t for t, i in vocab.items()}
    parents = {}
    for entry in tj["model"]["merges"]:
        a, b = merge_pair(entry)
        parents.setdefault(a + b, (a, b))                      # first (highest-priority) merge forms token
    keep = {t for t in vocab if len(t) == 1}                   # byte alphabet (256 symbols)
    stack = [by_id[i] for i in used_ids if i in by_id]
    while stack:
        t = stack.pop()
        if t in keep:                                          # seen (alphabet is pre-seeded, has no parents)
            continue
        keep.add(t)
        stack.extend(parents.get(t, ()))
    ids = {vocab[t] for t in keep} | {a["id"] for a in tj["added_tokens"]}
    return sorted(ids)
