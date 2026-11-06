# _expert_counts.py — validate MoE expert/topk counts before partition work.


def check_counts(experts, topk):
    if type(experts) is not int or experts <= 0:
        raise ValueError(f"experts {experts} must be a positive integer")
    if type(topk) is not int or not 0 < topk <= experts:
        raise ValueError(f"topk {topk} must be an integer in 1..{experts}")
