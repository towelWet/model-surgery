"""Refuse placeholder rows, textual tool-call imitation, and malformed structured calls."""
import json


def validate_training_data(path):
    count = calls = 0
    with open(path) as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("teacher_fill"):
                raise ValueError(f"row {line_number}: unfilled teacher placeholder")
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"row {line_number}: nonempty messages required")
            pending, seen = set(), set()
            for message in messages:
                content = message.get("content")
                if message.get("role") == "assistant" and isinstance(content, str) and "<tool_call>" in content:
                    raise ValueError(f"row {line_number}: chat-template tool-call text in content")  # Qwen XML imitation
                if isinstance(content, str):
                    try:
                        encoded = json.loads(content)
                    except (ValueError, TypeError):
                        encoded = None
                    if isinstance(encoded, list) and any(isinstance(v, dict) and "function" in v for v in encoded):
                        raise ValueError(f"row {line_number}: tool calls encoded in content")
                for call in message.get("tool_calls", []):
                    calls += 1
                    call_id = call.get("id")
                    if message.get("role") != "assistant" or not call_id or call_id in seen:
                        raise ValueError(f"row {line_number}: invalid tool call ID/role")
                    if call.get("type") != "function" or not isinstance(json.loads(call["function"]["arguments"]), dict):
                        raise ValueError(f"row {line_number}: invalid structured function call")
                    pending.add(call_id)
                    seen.add(call_id)
                if message.get("role") == "tool":
                    call_id = message.get("tool_call_id")
                    if call_id not in pending:
                        raise ValueError(f"row {line_number}: unmatched tool response")
                    pending.remove(call_id)
            if pending and not messages[-1].get("tool_calls"):
                raise ValueError(f"row {line_number}: unresolved tool call before final answer")
            count += 1
    if not count:
        raise ValueError("training data is empty")
    return {"rows": count, "structured_calls": calls}
