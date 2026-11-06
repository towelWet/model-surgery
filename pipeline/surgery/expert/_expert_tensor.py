# _expert_tensor.py — one checkpoint tensor after expert removal: [(new_key, tensor)] or [] if dropped.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
# Layouts handled (dim 0 = expert index in each):
#   per-expert   ...layers.L.mlp.experts.E.<proj>.<t>         drop / renumber E
#   stacked      ...layers.L.mlp.switch_mlp.<proj>.<t>  [E,…]  slice rows (Qwen3.5/3.6 MoE MLX layout)
#   router       ...layers.L.mlp.gate.<t>               [E,…]  slice rows (packed along last dim, so rows are safe)
# Old code knew only per-expert keys: on stacked checkpoints it shrank the router and kept all experts.
import re

import mlx.core as mx

PER_EXPERT = re.compile(r"(.*\.layers\.)(\d+)(\.mlp\.experts\.)(\d+)(\..*)")
ROW_SLICED = re.compile(r".*\.layers\.(\d+)\.mlp\.(?:switch_mlp\.\w+|gate)\.(?:weight|scales|biases|bias)$")


def expert_tensor(key, t, plan, n_experts):
    m = PER_EXPERT.match(key)
    if m:
        li, e = int(m.group(2)), int(m.group(4))
        drop = plan.get(li, set())
        if e in drop:
            return []
        new_e = e - sum(1 for r in drop if r < e)           # close the gaps left by dropped experts
        return [(f"{m.group(1)}{li}{m.group(3)}{new_e}{m.group(5)}", t)]
    m = ROW_SLICED.match(key)
    if m and int(m.group(1)) in plan:
        if t.shape[0] != n_experts:
            raise ValueError(f"{key}: dim0 {t.shape[0]} != num_experts {n_experts}")
        keep = [e for e in range(n_experts) if e not in plan[int(m.group(1))]]
        return [(key, t[mx.array(keep)])]
    return [(key, t)]                                       # attention, shared expert, norms …
