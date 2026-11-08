"""arg_validity denominator counts tool-expecting cases with no call (was: only replies that called a tool)."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import pipeline.eval.tool._loop as loop


def test_missed_call_counts_against_args():
    replies = iter([{"content": "sure", "tool_calls": []},           # tool expected, none called
                    {"tool_calls": [{"function": {"name": "bash", "arguments": '{"command": "cat /etc/hosts"}'}}]}])
    real = loop.chat
    loop.chat = lambda **_: next(replies)                             # fake server: no network
    try:
        sel, ok, scored = loop.run_cases([("Read /etc/hosts.", "read_file", "path"),
                                          ("Read /etc/hosts.", "read_file", "path", ["read_file", "bash"])])
    finally:
        loop.chat = real
    assert (sel, ok, scored) == (1, 1, 2)                             # miss is in denominator; bash cat accepted
