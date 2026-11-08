# _prompts.py — KL eval prompt list and constants; data only, no logic.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Disjoint from _calib.py CALIB and schema/_core.py CASES.

TOP_K = 32
MAX_NEW = 16                           # tokens per prompt — keeps runtime <10 min/model
import os
# Parent traces belong to ONE parent model: set MS_KL_CACHE per parent
# (e.g. db/<family>/<ver>/<size>/1/kl_parent.json). Was /tmp (wiped on reboot, shared by every parent).
CACHE = os.environ.get("MS_KL_CACHE",
                       os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "kl_parent_cache.json"))

# Disjoint from _calib.py CALIB and schema/_core.py CASES.
KL_PROMPTS = [
    "Describe how a bloom filter works.",
    "What is the difference between mutex and rwlock?",
    "Explain backpropagation in neural networks briefly.",
    "Why do hash tables degrade with poor hash functions?",
    "Summarize the CAP theorem tradeoffs.",
    "How does copy-on-write help in filesystem snapshots?",
    "What causes thundering herd in caches?",
    "Describe eventual consistency with an example.",
    "Explain how DNS recursive resolution works.",
    "What is vector clock used for?",
    "Compare merge sort and quicksort stability.",
    "How does Raft elect a leader?",
    "What is a memory barrier in concurrent programming?",
    "Explain idempotency in HTTP APIs.",
    "Why use B-trees for database indexes?",
    "Describe how TLS 1.3 reduces handshake latency.",
    "What is write amplification in SSDs?",
    "Explain the observer pattern.",
    "How does a generational garbage collector work?",
    "What is tail latency and p99?",
    "Describe map-reduce in one paragraph.",
    "Why normalize activations in deep networks?",
    "Explain Amdahl's law with an example.",
    "What is a happens-before relationship?",
    "How does LSM-tree compaction work?",
    "Describe double-entry bookkeeping.",
    "What is speculative execution in CPUs?",
    "Explain the two-phase commit protocol.",
    "How does a skip list achieve O(log n) search?",
]
