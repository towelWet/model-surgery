# _tools.py — TOOL_SCHEMA (OpenAI tools wire format) + per-case subsets for tool evals.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
# Schema data lives in tool_schema.json: JSON is the wire format the server receives verbatim,
# so it is the allowed exception to the TOML-first rule (edit the JSON, not this file).
import json
import os

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tool_schema.json")) as _f:
    TOOL_SCHEMA = json.load(_f)                             # read_file, bash, write_file

# Per-case schema subsets — distractor names test routing under a narrowed tool list.
TOOL_SCHEMA_READ_BASH = [t for t in TOOL_SCHEMA if t["function"]["name"] in ("read_file", "bash")]
TOOL_SCHEMA_WRITE_ONLY = [t for t in TOOL_SCHEMA if t["function"]["name"] == "write_file"]

# Similar-sounding distractors live in _distractor.py; only passed on cases that need them.
