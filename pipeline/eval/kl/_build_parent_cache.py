# _build_parent_cache.py — record parent's greedy top-k logprob trace per prompt.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Context shape MUST match _student.eval_student: user turn + ONE assistant turn
# holding the running prefix. Old builder appended each token as its own
# assistant message, so parent and student were conditioned on different
# chat-template renderings and KL compared mismatched contexts.
import json
import os

from pipeline.eval.kl._next_token import _next_token
from pipeline.eval.kl._prompts import CACHE, KL_PROMPTS, MAX_NEW


def build_parent_cache():
    """Run with PARENT model served. Rewrites CACHE after each prompt (partial = resumable view)."""
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    traces = []
    for prompt in KL_PROMPTS:
        prompt_msg = [{"role": "user", "content": prompt}]
        steps, prefix = [], ""
        for _ in range(MAX_NEW):
            msgs = prompt_msg + ([{"role": "assistant", "content": prefix}] if prefix else [])
            tok, topk = _next_token(msgs)                      # greedy token + parent top-k logprobs
            steps.append({"token": tok, "parent_lp": topk})
            prefix += tok                                      # same running prefix student will see
        traces.append({"prompt": prompt, "steps": steps})
        print(f"  cached {prompt[:50]}... ({len(steps)} tok)", flush=True)
        with open(CACHE, "w") as f:                            # direct overwrite (rename ops banned here)
            json.dump(traces, f)
    return traces
