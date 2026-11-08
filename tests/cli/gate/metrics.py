"""Malformed numeric evidence must fail even for report-only metrics."""
from pipeline.cli.gate._check_metric import check_metric


def test_nonfinite_and_nonnumeric_metrics_fail():
    for value in (float("nan"), float("inf"), "0.9", True):
        assert check_metric("s", "acc", {}, value, 1)[0] == "FAIL"
        assert check_metric("s", "acc", {"report_only": True}, 1, value)[0] == "FAIL"
    for value in (-1, float("nan"), "0.1", True):
        assert check_metric("s", "acc", {"max_drop": value}, 1, 1)[0] == "FAIL"
    assert check_metric("s", "acc", {"max_drop": 0.1}, 1, 0.95)[0] == "PASS"
