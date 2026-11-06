# _student.py — evaluate student model against parent logprob cache.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from _math import kl_topk
from _next_token import _next_token


def eval_student(traces):
    """Score student by KL divergence from parent traces."""
    total_kl, total_tok = 0.0, 0
    for trace in traces:
        prompt_msg = [{"role": "user", "content": trace["prompt"]}]
        prefix = ""
        for step in trace["steps"]:
            msgs = prompt_msg + ([{"role": "assistant", "content": prefix}] if prefix else [])
            _, student_lp = _next_token(msgs)
            total_kl += kl_topk(step["parent_lp"], student_lp)
            total_tok += 1
            prefix += step["token"]
    mean_kl = total_kl / total_tok if total_tok else float("nan")
    print(f"\nmean_kl_parent_student : {mean_kl:.4f}  ({total_tok} token positions)")
    return mean_kl
