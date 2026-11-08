# _grade.py — substring grader shared by code + rev suites; no I/O.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from pipeline.eval._strip_fences import strip_fences


def grade_response(text, need, any_of=None):
    """Pass if every `need` item appears and (if given) one `any_of` does.
    A need item that is a list is satisfied by any one of its spellings (e.g. ["jnz", "jne"])."""
    body = strip_fences(text).lower()                     # case-insensitive match
    if not body:
        return False                                      # empty reply never passes
    for s in need:                                        # all required terms
        alts = s if isinstance(s, list) else [s]          # list = equivalent spellings; correct paraphrase must pass
        if not any(x.lower() in body for x in alts):
            return False
    if any_of:
        return any(x.lower() in body for x in any_of)     # at least one alternative
    return True
