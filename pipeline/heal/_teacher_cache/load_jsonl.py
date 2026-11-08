# load_jsonl.py — read distill rows (one JSON conversation per line).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import json


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]     # skip blank lines
