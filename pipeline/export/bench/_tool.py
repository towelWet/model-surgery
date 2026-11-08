# _tool.py — tool-call slice of bench entry.py; no CLI.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
# Paths come from entry.py (add_eval_paths). Old line here inserted export/profile,
# a dir that never existed, so this module only imported when entry.py ran first.
from pipeline.profile._chat import chat
from pipeline.profile._eval_case import unpack_case
from pipeline.profile._validators import is_valid_json_args


def tool_metrics(cases):
    sel_ok = arg_ok = arg_scored = 0
    for case in cases:
        u = unpack_case(case)
        try:
            r = chat(messages=u["messages"], tools=u["tools"], max_tokens=200)
            calls = r.get("tool_calls", [])
            if u["exp_tool"] is None:
                if not calls and r.get("content", "").strip():
                    sel_ok += 1
                continue
            if not calls:
                continue
            name = calls[0]["function"]["name"]
            astr = calls[0]["function"].get("arguments", "")
            ok_sel = name in u["accept"]
            arg_key = u["req_arg"] if name == u["exp_tool"] else {
                "bash": "command", "read_file": "path", "write_file": "path",
            }.get(name, u["req_arg"])
            ok_arg = is_valid_json_args(astr, arg_key) if ok_sel else False
            sel_ok += ok_sel
            arg_scored += 1
            arg_ok += ok_arg
        except Exception:
            pass
    n = len(cases)
    return {
        "tool_selection_acc": round(sel_ok / n, 4),
        "arg_validity": round(arg_ok / arg_scored, 4) if arg_scored else None,
        "cases": n,
    }
