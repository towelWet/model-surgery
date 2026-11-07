"""DARE keeps ~density of the task vector, rescales by 1/density, and works in float32 in place."""
import numpy as np
from pipeline.surgery._merge_ties.merge_delta_np import merge_delta_np


def test_dare_drop_rescale_preserves_delta_in_expectation():
    base = np.zeros(200_000, dtype=np.float32)
    other = np.full_like(base, 2.0)                                   # task vector = 2 everywhere
    merged = merge_delta_np(base, other, 0.25, 1.0, 0.5, np.random.default_rng(7))
    kept = merged != 0
    assert merged.dtype == np.float32 and abs(kept.mean() - 0.25) < 0.01
    assert np.allclose(merged[kept], 0.5 * 2.0 / 0.25)                # lambda * delta / density
    assert abs(float(merged.mean()) - 1.0) < 0.02                     # E[merged] = lambda * delta
    assert base.max() == 0 and other.min() == 2.0                     # inputs untouched
    for bad in (0.0, 1.5, float("nan")):
        try:
            merge_delta_np(base, other, bad, 1.0, 1.0, np.random.default_rng(0))
        except ValueError:
            continue
        raise AssertionError(f"density {bad} accepted")
