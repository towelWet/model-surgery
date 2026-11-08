# _corpus_ids.py — every token id the tokenizer emits for given corpus text files.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from tokenizers import Tokenizer


def corpus_ids(tokenizer_json, corpus_files):
    tok = Tokenizer.from_file(str(tokenizer_json))
    used = set()
    for path in corpus_files:
        text = open(path, encoding="utf-8", errors="replace").read()
        for start in range(0, len(text), 200_000):             # chunk: huge single strings are slow to encode
            used.update(tok.encode(text[start:start + 200_000], add_special_tokens=False).ids)
    return used
