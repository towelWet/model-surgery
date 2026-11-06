# _stub_tool_call.py — offline assistant stub: JSON tool call matching user request shape.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import json


def stub_tool_response(kind, user):
    if "Read " in user:                        # "Read <path> and summarize." -> read_file
        path = user.split("Read ", 1)[1].split(" and")[0].strip()
        return json.dumps([{"function": {"name": "read_file", "arguments": json.dumps({"path": path})}}])
    if user.startswith("Run:"):                # "Run: <cmd>" -> bash
        cmd = user.split("Run:", 1)[1].strip()
        return json.dumps([{"function": {"name": "bash", "arguments": json.dumps({"command": cmd})}}])
    if user.startswith("Write "):              # "Write '<content>' to <path>." -> write_file
        rest = user.split("Write ", 1)[1]
        content, _, path_part = rest.partition(" to ")
        path = path_part.rstrip(".")
        content = content.strip().strip("'\"")  # undo repr() quoting from tool_request
        return json.dumps([{"function": {"name": "write_file", "arguments": json.dumps({"path": path, "content": content})}}])
    return json.dumps([{"function": {"name": "bash", "arguments": json.dumps({"command": "true"})}}])  # unknown shape: no-op call
