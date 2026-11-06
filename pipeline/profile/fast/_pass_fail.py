# _pass_fail.py — PASS/FAIL line for mean decode tok/s against threshold.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.


def fmt_result(tok_s, threshold):
    tag = "PASS" if tok_s >= threshold else "FAIL"
    return f"{tag}: mean decode {tok_s:.2f} tok/s >= {threshold} tok/s"
