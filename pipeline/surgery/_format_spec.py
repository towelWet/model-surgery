# _format_spec.py — collapse layer index set into compact 'a-b,c' text for logs.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.


def format_spec(indices):
    """Collapse a set of indices back into compact 'a-b,c' form for logging."""
    ordered = sorted(indices)
    runs, start, prev = [], None, None
    for i in ordered:
        if start is None:
            start = prev = i
        elif i == prev + 1:
            prev = i                                   # extend current run
        else:
            runs.append((start, prev))                 # close run, begin next
            start = prev = i
    if start is not None:
        runs.append((start, prev))
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)
