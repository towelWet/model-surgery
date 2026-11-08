# _speed_site_stage2.py — Tau polish pass for speed_site.sh (stage2 only).
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")))  # repo root: absolute pipeline.* / tests.* imports
from pipeline.profile.site.gen._api import complete
from pipeline.profile.site.gen._validate import score_html
from pipeline.heal.site._msgs import stage2_messages

draft_in, out = sys.argv[1], sys.argv[2]
max_t = int(os.environ.get("MAX_TOKENS2", "3072"))
with open(draft_in) as f:
    draft = f.read()
s2 = complete(stage2_messages(draft), max_tokens=max_t)
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
with open(out, "w") as f:
    f.write(s2["content"])
print(
    json.dumps(
        {
            "stage2_tokens": s2["completion_tokens"],
            "tok_per_s": round(s2["tok_per_s"], 2),
            "quality": score_html(s2["content"]),
        },
        indent=2,
    )
)
