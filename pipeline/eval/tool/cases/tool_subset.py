# tool_subset.py — only some tools offered for this case; data only.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from _tools import TOOL_SCHEMA_READ_BASH, TOOL_SCHEMA_WRITE_ONLY       # restricted tool lists

CASES = [
    {
        "messages": [{"role": "user", "content": "Run cal."}],
        "exp_tool": "bash", "req_arg": "command",
        "tools": TOOL_SCHEMA_READ_BASH,
    },
    {
        "messages": [{"role": "user", "content": "Read /etc/shells."}],
        "exp_tool": "read_file", "req_arg": "path",
        "tools": TOOL_SCHEMA_READ_BASH,
    },
    {
        "messages": [{"role": "user", "content": "Write ok to /tmp/onlywrite.txt."}],
        "exp_tool": "write_file", "req_arg": "path",
        "tools": TOOL_SCHEMA_WRITE_ONLY,
    },
    {
        "messages": [{"role": "user", "content": "Read /etc/motd."}],
        "exp_tool": "read_file", "req_arg": "path",
        "tools": TOOL_SCHEMA_READ_BASH,
    },
]
