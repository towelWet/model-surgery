"""Reject incomparable, incomplete, or diagnostic evidence before promotion."""


def check_evidence(suite, base, cand):
    rows = []
    for label, report in (("base", base), ("candidate", cand)):
        if report.get("suite") != suite:
            rows.append(("FAIL", f"{suite}: {label} suite identity missing or wrong"))
        if report.get("diagnostic_only") is not False:
            rows.append(("FAIL", f"{suite}: {label} lacks promotion-grade evidence"))
        if type(report.get("total")) is not int or report["total"] <= 0:
            rows.append(("FAIL", f"{suite}: {label} needs a positive case count"))
        if not report.get("evaluation_fingerprint"):
            rows.append(("FAIL", f"{suite}: {label} lacks evaluation fingerprint"))
    for key in ("total", "evaluation_fingerprint", "grader"):
        if base.get(key) != cand.get(key):
            rows.append(("FAIL", f"{suite}: incomparable {key}"))
    return rows
