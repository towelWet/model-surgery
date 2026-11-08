# _stub_tool_call.py — offline assistant stub: structured tool call matching user request shape.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import json


def stub_tool_response(kind, user):
    if "Read " in user:                        # "Read <path> and summarize." -> read_file
        path = user.split("Read ", 1)[1].split(" and")[0].strip()
        return [{"id": "call_stub", "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": path})}}]
    if user.startswith("Run:"):                # "Run: <cmd>" -> bash
        cmd = user.split("Run:", 1)[1].strip()
        return [{"id": "call_stub", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": cmd})}}]
    if user.startswith("Write "):              # "Write '<content>' to <path>." -> write_file
        rest = user.split("Write ", 1)[1]
        content, _, path_part = rest.partition(" to ")
        path = path_part.rstrip(".")
        content = content.strip().strip("'\"")  # undo repr() quoting from tool_request
        return [{"id": "call_stub", "type": "function", "function": {"name": "write_file", "arguments": json.dumps({"path": path, "content": content})}}]
    raise ValueError("unknown synthetic tool request shape")
