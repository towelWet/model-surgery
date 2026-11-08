# rev/_expand.py — expand topic templates into training rows; no CLI.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# WARNING (2026-11-06 audit): assistant turns are PLACEHOLDERS ("pass  # teacher
# distill target", "concise expert answer for <tag>"). Rows need a teacher fill
# before training; training on them as-is teaches stubs.
from pipeline.heal.code.rev._topics import ALGOS, PATS, PY_TASKS, REV_TASKS, WEB_TASKS
from pipeline.heal.code._chat_row import _row


def expand_rows(rng, n):
    rows = []
    i = 0
    while len(rows) < n:
        kind = i % 3                                  # round-robin python / web / rev
        if kind == 0:
            tpl, stub, tag = rng.choice(PY_TASKS)
            algo, pat = rng.choice(ALGOS), rng.choice(PATS)
            user = tpl.format(algo=algo, pat=pat, feat="debounce")
            asst = stub.format(algo=algo, pat=pat) + "\n    pass  # teacher distill target"
        elif kind == 1:
            tpl, stub, tag = rng.choice(WEB_TASKS)
            user = tpl.format(comp="pricing card", feat="modal toggle")
            asst = f"{stub}\n<!-- scaffold for {tag} -->"
        else:
            tpl, stub, tag = rng.choice(REV_TASKS)
            user = tpl.format(hex="0f 05", name=".text", fn="parse_header")
            asst = f"{stub}: concise expert answer for {tag}."
        rows.append(_row(user, asst, tag))
        i += 1
    return rows
