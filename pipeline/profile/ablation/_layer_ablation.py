# _layer_ablation.py — score one ablation setting (baseline vs zeroed layers).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from pipeline.profile.ablation.nll._core import mean_nll
from pipeline.profile._mlx_load import load


def layer_ablation(model_dir, prompts, zero_layers=None):
    """Score one ablation setting; loads model each call (caller may cache)."""
    model, tok = load(model_dir)
    baseline = mean_nll(model, tok, prompts, skip_layers=None)               # full-model NLL
    ablated = mean_nll(model, tok, prompts, skip_layers=zero_layers or None)  # zeroed-group NLL
    return {
        "baseline_nll": baseline,
        "ablated_nll": ablated,
        "loss_delta": ablated - baseline,                                     # positive = damage
        "zero_layers": zero_layers or [],
    }
