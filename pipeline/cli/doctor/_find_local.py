# _find_local.py — resolve dotted bare-name import against search dirs: module file or package __init__.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import os


def find_local(dotted, dirs):
    rel = dotted.replace(".", os.sep)
    for d in dirs:
        for cand in (os.path.join(d, rel + ".py"), os.path.join(d, rel, "__init__.py")):
            if os.path.isfile(cand):
                return cand
        if os.path.isdir(os.path.join(d, rel)):
            return os.path.join(d, rel)                     # namespace package dir (no __init__)
    return None
