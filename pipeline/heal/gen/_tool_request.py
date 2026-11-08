# _tool_request.py — random tool request (user text + schema) for distillation JSONL; no CLI.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Assistant stubs moved to _stub_tool_call.py (re-exported below).
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
from pipeline.profile.schema._core import TOOL_SCHEMA

from pipeline.heal.gen.corpus._data import BASH_CMDS, READ_PATHS, WRITE_PAIRS
from pipeline.heal.gen._stub_tool_call import stub_tool_response  # re-exported for _corpus_rows


def tool_request(rng):
    kind = rng.choice(["read", "bash", "write"])
    if kind == "read":
        p = rng.choice(READ_PATHS)
        return f"Read {p} and summarize.", TOOL_SCHEMA
    if kind == "bash":
        c = rng.choice(BASH_CMDS)
        return f"Run: {c}", TOOL_SCHEMA
    path, content = rng.choice(WRITE_PAIRS)
    return f"Write {content!r} to {path}.", TOOL_SCHEMA
