# distractor.py — look-alike tool names offered (read_file vs file_read); data only.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from pipeline.profile.schema._distractor import TOOL_SCHEMA_DISTRACTOR       # look-alike tool names

CASES = [
    {
        "messages": [{"role": "user", "content": "Run echo test via shell."}],
        "exp_tool": "bash", "req_arg": "command",
        "tools": TOOL_SCHEMA_DISTRACTOR,
        "accept": ["bash"],
    },
    {
        "messages": [{"role": "user", "content": "Open /etc/hostname with read_file not file_read."}],
        "exp_tool": "read_file", "req_arg": "path",
        "tools": TOOL_SCHEMA_DISTRACTOR,
        "accept": ["read_file"],
    },
    {
        "messages": [{"role": "user", "content": "Read /etc/hosts using the file tool."}],
        "exp_tool": "read_file", "req_arg": "path",
        "tools": TOOL_SCHEMA_DISTRACTOR,
        "accept": ["read_file"],
    },
]
