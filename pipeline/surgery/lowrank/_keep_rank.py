# _keep_rank.py — rank whose two factors hold `ratio` of original matrix parameters.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# r·(out + in) = ratio·out·in. Rounded DOWN to `group` so A's last dim (r) still packs into
# MLX affine quantization groups; never below one group.


def keep_rank(out, inn, ratio, group=64):
    r = int(ratio * out * inn / (out + inn))
    return max(group, r // group * group)
