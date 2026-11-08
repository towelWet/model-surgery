# _py_files.py — every tracked-looking .py under pipeline/ (skips caches and quarantined trees).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import pathlib

SKIP_PARTS = {"__pycache__", ".grit", "cache"}              # generated or data dirs, never source


def py_files(root):
    return [p for p in sorted(pathlib.Path(root).rglob("*.py"))
            if not SKIP_PARTS & set(p.relative_to(root).parts)]
