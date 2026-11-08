# gen/_task.py — load site-gen bench task from site_task.toml; exposes old constant names.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
# Strings used to live here as Python literals; data now in TOML (edit site_task.toml, not this file).
import os
import tomllib

from pipeline.profile.site._merge_scaffold import merge_scaffold as _merge

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "site_task.toml"), "rb") as _f:
    _T = tomllib.load(_f)

REFERENCE_BRIEF = _T["reference_brief"]
SYSTEM_PROMPT = _T["system_prompt"]
SCAFFOLD_FILL_SYSTEM = _T["scaffold_fill_system"]
SCAFFOLD = _T["scaffold"]
SCAFFOLD_MARKER = _T["scaffold_marker"]


def merge_scaffold(fragment):                               # old one-arg signature kept for callers
    return _merge(fragment, SCAFFOLD, SCAFFOLD_MARKER)
