# litter.py — doctor check: .DS_Store, stray .pyc outside __pycache__, editor/backup junk in skill tree.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import os
import pathlib

JUNK = (".DS_Store", ".orig", ".rej", ".bak", "~", ".swp")


def run(root):
    skill = pathlib.Path(root).parent                       # whole skill, not just pipeline/
    out = []
    for p in skill.rglob("*"):
        if ".grit" in p.parts or not p.is_file():
            continue
        if p.name.endswith(JUNK) or (p.suffix == ".pyc" and "__pycache__" not in p.parts):
            out.append(os.path.relpath(p, skill))
    return out


CHECK = {"name": "litter", "help": "no .DS_Store / stray .pyc / backup files", "run": run}
