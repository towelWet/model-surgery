# _tool_loop.py — score tool-call cases (selection + argument validity); no CLI.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from _chat import chat
from _eval_case import unpack_case
from _validators import is_valid_json_args


def run_cases(cases):
    sel_ok = arg_ok = arg_scored = 0
    for case in cases:
        u = unpack_case(case)
        label, messages, tools = u["label"], u["messages"], u["tools"]
        exp_tool, req_arg, accept = u["exp_tool"], u["req_arg"], u["accept"]
        try:
            resp = chat(messages=messages, tools=tools, max_tokens=200)
            calls = resp.get("tool_calls", [])
            if exp_tool is None:
                if not calls and resp.get("content", "").strip():
                    sel_ok += 1
                    print(f"  sel=OK arg=OK | {label!r} -> (no tool, plain answer)")
                elif not calls:
                    print(f"  sel=FAIL arg=FAIL | {label!r} -> empty response, no tool")
                else:
                    name = calls[0]["function"]["name"]
                    print(f"  sel=FAIL arg=FAIL | {label!r} -> unexpected {name}")
                continue
            if not calls:
                print(f"  MISS  | {label!r} -> no tool call")
                continue
            name = calls[0]["function"]["name"]
            args = calls[0]["function"].get("arguments", "")
            ok_sel = name in accept
            arg_key = req_arg if name == exp_tool else {
                "bash": "command", "read_file": "path", "write_file": "path",
            }.get(name, req_arg)
            ok_arg = is_valid_json_args(args, arg_key) if ok_sel else False
            sel_ok += ok_sel
            arg_scored += 1
            arg_ok += ok_arg
            sel_str = "OK" if ok_sel else "FAIL"
            arg_str = "OK" if ok_arg else ("n/a" if not ok_sel else "FAIL")
            print(f"  sel={sel_str} arg={arg_str} | {label!r} -> {name}")
        except Exception as e:
            print(f"  ERROR | {label!r} -> {e}")
    return sel_ok, arg_ok, arg_scored
