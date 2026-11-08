# _write_slow_files.py — vocab.json + merges.txt matching trimmed tokenizer.json.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Slow (Python) tokenizers read these instead of tokenizer.json; copied originals would
# carry dropped ids past new vocab_size, so they are rewritten, never left stale.
import json
import os
from pipeline.surgery.vocab._merge_pair import merge_pair


def write_slow_files(dest, tj):
    if os.path.exists(os.path.join(dest, "vocab.json")):
        json.dump(tj["model"]["vocab"], open(os.path.join(dest, "vocab.json"), "w"), ensure_ascii=False)
    if os.path.exists(os.path.join(dest, "merges.txt")):
        lines = ["#version: 0.2"] + [" ".join(merge_pair(m)) for m in tj["model"]["merges"]]
        open(os.path.join(dest, "merges.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
