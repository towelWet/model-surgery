# _sum_distances.py — per prompt: capture hidden states once, add angular distance for every block cut.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import mlx.core as mx

from _angular import angular_distance
from _hidden import capture


def sum_distances(model, tok, prompts, starts, max_tokens):
    totals = {b: {s: 0.0 for s in ss} for b, ss in starts.items()}
    for n, prompt in enumerate(prompts, 1):
        ids = mx.array(tok.encode(prompt)[:max_tokens]).reshape(1, -1)  # truncate: caps peak RAM
        states = capture(model, ids)                        # reused across every block size
        for b, ss in starts.items():
            for s in ss:
                totals[b][s] += angular_distance(states[s], states[s + b])  # input vs output of cut block
        del states
        mx.clear_cache()                                    # release Metal buffers between prompts
        print(f"  [{n}/{len(prompts)}] {prompt[:46]}")
    return totals
