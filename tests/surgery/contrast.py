"""Opposite signed activations of equal magnitude must not collapse to zero contrast."""
from pipeline.surgery.neuron._contrast import compute_contrast


def test_contrast_uses_signed_activation_means():
    target = [{"activations": {"0": [-2, 3]}}]
    baseline = [{"activations": {"0": [2, -3]}}]
    result = compute_contrast(target, baseline)
    assert result[0]["neuron"] == 1 and result[0]["contrast"] == 6
    assert result[1]["neuron"] == 0 and result[1]["contrast"] == -4
