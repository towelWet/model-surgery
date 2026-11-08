"""Greedy soup starts from best model and keeps a candidate only when soup score improves."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import sys
from pathlib import Path


def test_greedy_soup_keeps_only_improvements():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root: absolute pipeline.* / tests.* imports
    from pipeline.surgery.soup.greedy._core import greedy_soup
    alone = {"a": 2.0, "b": 1.0, "c": 3.0}
    made = {}
    merge = made.__setitem__                                 # record which models each trial averaged
    soup_score = {("b", "a"): 0.8, ("b", "a", "c"): 0.9}     # adding c makes soup worse
    score = lambda m: alone[m] if m in alone else soup_score[tuple(made[m])]
    soup, best, best_dir, log = greedy_soup(["a", "b", "c"], "W", lambda ms, out: merge(out, ms), score)
    assert soup == ["b", "a"] and best == 0.8 and best_dir.endswith("trial-01")
    assert log[2]["soup_with"] == 0.9                        # c tried and rejected
