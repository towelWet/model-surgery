"""Validate case identities and expected tools before spending inference time."""
import json


def check_cases(cases):
    errors, seen, ids = [], set(), set()
    for number, case in enumerate(cases, 1):
        key = json.dumps(case, sort_keys=True)
        if key in seen:
            errors.append(f"case {number}: duplicate case")
        seen.add(key)
        if isinstance(case, dict) and "id" in case:
            if case["id"] in ids:
                errors.append(f"case {number}: duplicate id {case['id']}")
            ids.add(case["id"])
        if isinstance(case, dict) and "tools" in case:
            offered = {tool["function"]["name"] for tool in case["tools"]}
            accepted = case.get("accept", [case.get("exp_tool")])
            if case.get("exp_tool") is not None and not offered.intersection(accepted):
                errors.append(f"case {number}: expected tool is not offered")
    if not cases:
        errors.append("empty suite")
    return errors
