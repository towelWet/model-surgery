# _site_result.py — bench result record: speed, token counts, HTML sanity score.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from _gen_validate import score_html

TARGETS = {300: (33, 67), 600: (17, 33)}  # sec -> (tok/s for 10k, 20k)


def site_result(args, run, html):
    return {
        "max_tokens": args.max_tokens,
        "scaffold": args.scaffold,
        "elapsed_s": round(run["elapsed_s"], 2),
        "completion_tokens": run["completion_tokens"],
        "tok_per_s": round(run["tok_per_s"], 2),
        "usage": run["usage"],
        "quality": score_html(html),
        "merged_chars": len(html),
        "targets_5min_tok_s": TARGETS[300],                 # tok/s needed to finish 10k/20k tokens in 5 min
    }
