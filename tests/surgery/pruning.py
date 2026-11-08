"""Refuse invalid fractions, expert permutations and layer identifiers."""
from pipeline.surgery.expert._removal_plan import removal_plan


def test_expert_pruning_inputs_are_strict():
    usage = {"per_layer": {"0": {"prune_order": [3, 1, 2, 0]}}}
    plan, count = removal_plan(usage, 4, 0.5)
    assert plan == {0: {3, 1}} and count == 2
    bad = [(usage, f) for f in (0, -1, 1, 2, float("nan"), float("inf"), 0.01)]
    bad += [({"0": [{"expert": e} for e in order]}, 0.5)
            for order in ([0, 0, 1, 2], [-1, 0, 1, 2], [0, 1, 2, 4], [True, 1, 2, 3])]
    bad += [({}, 0.5), ({"-1": [{"expert": e} for e in range(4)]}, 0.5)]
    for records, frac in bad:
        try:
            removal_plan(records, 4, frac)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid pruning input: {records}, {frac}")
