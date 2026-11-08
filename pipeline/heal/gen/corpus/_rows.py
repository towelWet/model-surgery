# corpus/_rows.py — build distill JSONL rows; no CLI.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import json

from pipeline.heal.gen._tool_request import stub_tool_response, tool_request  # was _corpus (collided with heal/_corpus)
from pipeline.heal.gen.corpus._data import GENERAL, SYSTEM


def build_rows(rng, n_tool, n_gen, prompts_only, chat_fn=None):
    rows = []
    for i in range(n_tool):
        user, tools = tool_request(rng)
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        if prompts_only:
            response = {"role": "assistant", "content": None,
                        "tool_calls": stub_tool_response("tool", user)}
        else:
            resp = chat_fn(messages, tools=tools, max_tokens=256)
            response = {"role": "assistant", "content": resp.get("content")}
            if resp.get("tool_calls"):
                response["tool_calls"] = resp["tool_calls"]
        messages.append(response)
        rows.append({"messages": messages, "tools": tools, "teacher_fill": prompts_only})
        if i % 200 == 0:
            print(f"  tool {i}/{n_tool}")
    for i in range(n_gen):
        user = rng.choice(GENERAL)
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        if prompts_only:
            assistant = "[TEACHER_FILL: concise answer expected]"
        else:
            resp = chat_fn(messages, tools=None, max_tokens=256)
            assistant = resp.get("content", "")
        messages.append({"role": "assistant", "content": assistant})
        rows.append({"messages": messages, "teacher_fill": prompts_only})
        if i % 200 == 0:
            print(f"  gen {i}/{n_gen}")
    return rows
