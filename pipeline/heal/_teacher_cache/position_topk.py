# position_topk.py — teacher top-K ids + logprobs for one assistant position, padded to TOP_K.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from ._constants import TOP_K
from .token_id import token_id
from .topk_at_prefix import topk_at_prefix


def position_topk(tok, messages, tools, full_ids, start, pos):
    # Teacher-forced prefix up to (not including) label token.
    prefix_msgs = messages[:-1]
    asst_so_far = tok.decode(full_ids[start:pos], skip_special_tokens=False)
    if asst_so_far:
        prefix_msgs = prefix_msgs + [{"role": "assistant", "content": asst_so_far}]  # partial assistant turn
    _, topk = topk_at_prefix(prefix_msgs, tools)
    ids_k, probs_k = [], []
    for tok_str, logp in sorted(topk.items(), key=lambda x: -x[1])[:TOP_K]:  # most likely first
        tid = token_id(tok, tok_str)
        if tid is None:
            continue                                       # non-round-tripping string: drop
        ids_k.append(tid)
        probs_k.append(logp)
    while len(ids_k) < TOP_K:
        ids_k.append(0)                                    # pad id 0 with ~zero prob
        probs_k.append(-1e9)
    return ids_k[:TOP_K], probs_k[:TOP_K]
