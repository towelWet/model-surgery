# run/_text_suite.py — shared body of code/run.py and rev/run.py.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from pipeline.eval._load_cases import load_cases
from pipeline.eval._suite_args import suite_args
from pipeline.eval._write_report import write_report


def run_text_suite(suite, suite_dir, system, max_tokens, description):
    """Load cases, list or run them, print accuracy, optionally write TOML report."""
    a = suite_args(description)
    cases, by_cat = load_cases(suite_dir, a.only)
    if a.list:                                                   # dry inspection: no server needed
        for cat, cs in by_cat.items():
            print(f"{cat:14} {len(cs)}")
        return 0
    from pipeline.eval._text_loop import run_text_cases                        # imported late: --list works without _chat deps
    ok, results = run_text_cases(cases, system, max_tokens)
    status = dict(results)
    per = {cat: {"passed": sum(status[c["id"]] == "OK" for c in cs), "total": len(cs)} for cat, cs in by_cat.items()}
    n = len(cases)
    print(f"\n{suite}_acc : {ok}/{n} = {ok / n:.1%}")
    if a.out:
        write_report(a.out, suite, a.model, {"passed": ok, "total": n, "acc": ok / n}, per)
        print(f"report -> {a.out}")
    return 0
