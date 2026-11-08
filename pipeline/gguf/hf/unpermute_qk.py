# unpermute_qk.py — undo llama.cpp's Llama q/k rope row permutation (GGUF -> HF layout).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# convert_hf_to_gguf.py LlamaModel.permute writes, per head, rows reordered as
#   w.reshape(h, 2, d/2, ...).swapaxes(1, 2)   (HF half-split rope -> interleaved pairs)
# so GGUF rows are viewed (h, d/2, 2, ...); swapping axes 1, 2 back restores HF order.
import mlx.core as mx


def unpermute_qk(w, n_head):
    """w [n_head*d, ...] as stored in GGUF -> HF row order. q: n_head = head_count; k: head_count_kv."""
    shape = w.shape
    return mx.reshape(mx.swapaxes(mx.reshape(w, (n_head, shape[0] // n_head // 2, 2, *shape[1:])), 1, 2), shape)
