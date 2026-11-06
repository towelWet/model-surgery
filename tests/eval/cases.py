"""Impossible tools, duplicate data, empty suites, and duplicate ids are rejected."""
from pipeline.eval.inspection.cases import check_cases


def test_case_validation_rejects_invalid_suites():
    impossible = {"messages": [], "exp_tool": "read", "tools": [{"function": {"name": "write"}}]}
    assert check_cases([impossible])
    assert check_cases([("same", "tool", "arg")] * 2)
    assert check_cases([{"id": "same", "prompt": "one"}, {"id": "same", "prompt": "two"}])
    assert check_cases([])
    assert not check_cases([{"messages": [], "exp_tool": "read", "tools": [{"function": {"name": "read"}}]}])
