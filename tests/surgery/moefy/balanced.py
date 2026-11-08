"""Balanced k-means recovers planted co-activation groups with exactly equal expert sizes."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import sys
from pathlib import Path
import numpy as np


def test_balanced_kmeans_recovers_planted_groups():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root: absolute pipeline.* / tests.* imports
    from pipeline.surgery.moefy.balanced._kmeans import balanced_kmeans
    rng = np.random.default_rng(1)
    topic = rng.integers(0, 4, 400)                          # each token belongs to one of 4 topics
    acts = rng.normal(0, 0.05, (400, 32))                    # background noise
    for g in range(4):                                       # neurons 8g..8g+7 fire on topic g only
        acts[topic == g, 8 * g:8 * g + 8] += 1.0
    labels = balanced_kmeans(acts, 4)
    assert sorted(np.bincount(labels)) == [8, 8, 8, 8]       # equal widths: stackable experts
    for g in range(4):
        assert len(set(labels[8 * g:8 * g + 8])) == 1        # each planted group lands in one expert
