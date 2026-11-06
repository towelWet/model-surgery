# multi_turn.py — tool choice depends on earlier turns; data only.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.

CASES = [
    {
        "messages": [
            {"role": "user",      "content": "Prepare a scratch file for me."},
            {"role": "assistant", "content": "What path and contents?"},
            {"role": "user",      "content": "Path /tmp/scratch.txt with content scratch."},
        ],
        "exp_tool": "write_file", "req_arg": "path",
    },
    {
        "messages": [
            {"role": "user",      "content": "I want to check system load."},
            {"role": "assistant", "content": "Should I run a command?"},
            {"role": "user",      "content": "Yes, run uptime."},
        ],
        "exp_tool": "bash", "req_arg": "command",
    },
    {
        "messages": [
            {"role": "user",      "content": "I need to inspect a config file."},
            {"role": "assistant", "content": "Which path should I read?"},
            {"role": "user",      "content": "Use /etc/hosts."},
        ],
        "exp_tool": "read_file", "req_arg": "path",
    },
]
