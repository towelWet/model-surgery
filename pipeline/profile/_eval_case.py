# _eval_case.py — unpack one tool_eval CASES entry; shape only, no I/O.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from _schema import TOOL_SCHEMA


def unpack_case(case):
    """Return dict with messages, exp_tool, req_arg, accept, tools, label."""
    if isinstance(case, dict):
        messages = case["messages"]
        exp_tool = case.get("exp_tool")
        req_arg = case.get("req_arg")
        accept = case.get("accept", [exp_tool] if exp_tool else [])
        tools = case.get("tools", TOOL_SCHEMA)  # no_tool cases get tools too: declining must be a choice (was None = free pass)
        label = messages[-1]["content"][:60]
        return {
            "messages": messages,
            "exp_tool": exp_tool,
            "req_arg": req_arg,
            "accept": accept,
            "tools": tools,
            "label": label,
        }
    if len(case) == 4:
        user_msg, exp_tool, req_arg, accept = case
        return {
            "messages": [{"role": "user", "content": user_msg}],
            "exp_tool": exp_tool,
            "req_arg": req_arg,
            "accept": accept,
            "tools": TOOL_SCHEMA,
            "label": user_msg,
        }
    user_msg, exp_tool, req_arg = case
    return {
        "messages": [{"role": "user", "content": user_msg}],
        "exp_tool": exp_tool,
        "req_arg": req_arg,
        "accept": [exp_tool],
        "tools": TOOL_SCHEMA,
        "label": user_msg,
    }
