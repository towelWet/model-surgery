# _bench_speed.py — one timed generation for export bench; returns rough speed numbers.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# HONEST LABELS: counts whitespace words, not tokens, and times whole request
# (not time-to-first-token). Keys keep old names (tok_per_s, ttft_s) so existing
# baseline JSON files still diff; treat both as approximate.
import time

from _chat import chat

PROMPT = "Write a Python function that reads a config file and returns a dict."


def measure_speed():
    """Return {'tok_per_s': words/s, 'ttft_s': total request seconds}."""
    print("Measuring generation speed...")
    t0 = time.perf_counter()
    resp = chat([{"role": "user", "content": PROMPT}], max_tokens=128)
    elapsed = time.perf_counter() - t0                          # whole request, not first token
    n_words = len(resp.get("content", "").split())               # words, not tokenizer tokens
    print(f"  {elapsed:.2f}s  {n_words} words  {n_words / elapsed:.1f} words/s")
    return {"tok_per_s": round(n_words / elapsed, 2), "ttft_s": round(elapsed, 3)}
