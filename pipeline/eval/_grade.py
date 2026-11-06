# _grade.py — substring grader shared by code + rev suites; no I/O.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from _strip_fences import strip_fences


def grade_response(text, need, any_of=None):
    """Pass if every `need` substring appears and (if given) one `any_of` does."""
    body = strip_fences(text).lower()                     # case-insensitive match
    if not body:
        return False                                      # empty reply never passes
    if any(s.lower() not in body for s in need):          # all required terms
        return False
    if any_of:
        return any(x.lower() in body for x in any_of)     # at least one alternative
    return True
