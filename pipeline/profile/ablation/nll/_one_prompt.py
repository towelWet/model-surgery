# nll/_one_prompt.py — teacher-forced mean NLL of one prompt (optionally with layers skipped).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import mlx.core as mx

from pipeline.profile.ablation._forward import forward_logits


def nll_one_prompt(model, tokenizer, text, skip_layers=None):
    ids = tokenizer.encode(text)
    if len(ids) < 2:
        return None                                         # need >=1 next-token target
    x = mx.array(ids).reshape(1, -1)                        # [1, seq]
    logits = forward_logits(model, x, skip_layers=skip_layers)
    log_p = logits - mx.logsumexp(logits, axis=-1, keepdims=True)  # log-softmax over vocab
    tgt = x[:, 1:]                                          # position t predicts token t+1
    picked = mx.take_along_axis(log_p[:, :-1, :], tgt[:, :, None], axis=-1).squeeze(-1)
    return float(-mx.mean(picked))
