"""Reject changed suites and legacy/diagnostic scores even when accuracy is high."""
from pipeline.cli.gate.evidence import check_evidence


def test_incomparable_and_diagnostic_evidence_blocked():
    base = dict(suite="test", total=2, evaluation_fingerprint="fixture", grader="fixture",
                diagnostic_only=False)
    assert not check_evidence("test", base, base.copy())
    for key, value in (("total", 3), ("evaluation_fingerprint", "changed"),
                       ("grader", "changed"), ("diagnostic_only", True), ("suite", "other")):
        candidate = {**base, key: value}
        assert check_evidence("test", base, candidate), key
    assert check_evidence("test", {}, {})
