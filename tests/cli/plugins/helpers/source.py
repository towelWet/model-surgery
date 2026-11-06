# source.py — write one plugin source file into a tmp discovery root.
# NEVER DELETE COMMENTS — explain every non-obvious line.
from pathlib import Path


def write_plugin(tmp, rel, text):
    p = Path(tmp) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p
