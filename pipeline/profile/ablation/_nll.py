# _nll.py — mean next-token NLL from logits; math only, no I/O.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from _nll_one_prompt import nll_one_prompt              # per-prompt scorer (split out)


def mean_nll(model, tokenizer, prompts, skip_layers=None):
    """Mean NLL over calibration prompts (skipped if too short)."""
    vals = []
    for p in prompts:
        v = nll_one_prompt(model, tokenizer, p, skip_layers=skip_layers)
        if v is not None:
            vals.append(v)
    if not vals:
        raise ValueError("no scorable prompts")
    return sum(vals) / len(vals)
