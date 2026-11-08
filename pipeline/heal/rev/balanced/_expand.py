# balanced/_expand.py — merge extracted rows + synthetic rev; no CLI.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from pipeline.heal.rev.balanced._data import CODE_TAGS, REV_TAGS, SYSTEM
from pipeline.heal.rev._load_tagged import load_tagged
from pipeline.heal.rev._synth_rev import _synth_rev


def build_balanced(rng, src_jsonl, total=3000, rev_ratio=0.6):
    """Build balanced rev + code training corpus."""
    n_rev = int(total * rev_ratio)
    n_code = total - n_rev
    rev_rows = load_tagged(src_jsonl, REV_TAGS)
    code_rows = load_tagged(src_jsonl, CODE_TAGS)
    rng.shuffle(rev_rows)
    rng.shuffle(code_rows)
    rev_pick = rev_rows[:n_rev]
    while len(rev_pick) < n_rev:                         # pad with synthetic if short
        rev_pick.append(_synth_rev(rng))
    code_pick = code_rows[:n_code]
    while len(code_pick) < n_code and code_rows:         # pad by resampling
        code_pick.append(rng.choice(code_rows))
    rows = rev_pick + code_pick
    rng.shuffle(rows)
    for r in rows:
        r["messages"][0]["content"] = SYSTEM              # inject system prompt
    return rows
