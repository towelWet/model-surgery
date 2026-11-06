# _teacher_cache — teacher top-32 logit harvest for KD heal; no CLI (driver: heal/cache_teacher_logits.py).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from ._constants import DEFAULT_XMEM, FALLBACK_CACHE, TOP_K
from .apply_template import apply_template
from .assistant_span import assistant_span
from .cache_root import cache_root
from .harvest import harvest
from .load_jsonl import load_jsonl
from .position_weight import position_weight
from .token_id import token_id
from .topk_at_prefix import topk_at_prefix
from .swap_server import swap_server
from .tokenizer_path import tokenizer_path
from .cache_args import cache_args
