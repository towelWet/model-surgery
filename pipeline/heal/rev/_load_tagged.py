# _load_tagged.py — load JSONL rows filtered by tag set.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import json


def load_tagged(path, tags):
    """Load JSONL rows whose 'tag' field is in the given set."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            if o.get("tag") in tags:
                out.append(o)
    return out
