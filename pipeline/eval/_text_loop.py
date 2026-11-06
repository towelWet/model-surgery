# _text_loop.py — run graded free-text cases (code + rev suites) against live server.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Replaces two copy-pasted loops (code/_eval_loop.py, rev/_eval_loop.py) that
# differed only in system prompt and default max_tokens.
from _chat import chat                                    # tensor-array OpenAI-style client
from _grade import grade_response


def run_text_cases(cases, system, default_max_tokens):
    """Return (passed, results) where results = [(id, 'OK'|'FAIL'|'ERROR')]."""
    ok, results = 0, []
    for c in cases:
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": c["prompt"]}]
        try:
            resp = chat(messages=msgs, tools=None,
                        max_tokens=c.get("max_tokens", default_max_tokens))
            passed = grade_response(resp.get("content", "") or "", c["need"], c.get("any_of"))
            ok += passed
            mark = "OK" if passed else "FAIL"
        except Exception as e:                            # server down / bad reply: count as fail, keep going
            mark = f"ERROR {e}"
        results.append((c["id"], mark))
        print(f"  {mark} | {c['id']}")
    return ok, results
