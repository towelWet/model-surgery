# _check_metric.py — apply one gate rule to (base, cand) values → (status, message).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import math


def check_metric(suite, name, rule, base, cand):
    label = f"{suite}.{name}"
    for value in (base, cand):
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
            return "FAIL", f"{label}: metric must be a finite number"
    for key in ("max_drop", "max_rise"):
        value = rule.get(key, 0.0)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            return "FAIL", f"{label}: {key} must be finite and nonnegative"
    if rule.get("report_only") and (cand is None or base is None):
        return "INFO", f"{label}: not run (report-only rule)"     # kl is skipped by default in ms eval
    if cand is None:
        return "FAIL", f"{label}: missing in candidate reports"   # missing evidence never passes
    if base is None:
        return "FAIL", f"{label}: missing in base reports (run ms eval on base model first)"
    delta = cand - base
    text = f"{label}: base {base:.4f} cand {cand:.4f} ({delta:+.4f})"
    if rule.get("report_only"):
        return "INFO", text
    if rule.get("higher", True):
        ok = cand >= base - rule.get("max_drop", 0.0)
    else:
        ok = cand <= base + rule.get("max_rise", 0.0)
    return ("PASS" if ok else "FAIL"), text
