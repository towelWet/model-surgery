# _distractor.py — distractor tool list; data only.
TOOL_SCHEMA_DISTRACTOR = [
    {
        "type": "function",
        "function": {
            "name":        "read_file",
            "description": "Read a file from disk",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "file_read",          # distractor — similar name, wrong tool
            "description": "Read file contents (legacy alias)",
            "parameters": {"type": "object", "properties": {"filepath": {"type": "string"}}, "required": ["filepath"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "bash",
            "description": "Run a shell command",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name":        "shell_exec",          # distractor
            "description": "Execute shell (deprecated)",
            "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]},
        },
    },
]

# Each entry: tuple (user, exp_tool, req_arg) or 4-tuple with accept list, or dict.
# Held-out from _calib.py CALIB — do not reuse calibration prompts here.
