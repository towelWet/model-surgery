# LoRA self-distillation — cheap heal on attention + shared, experts frozen

Source: self-distillation-after-pruning practice (2026); SlimQwen (May 2026) finding
that KD + LM loss beats KD alone on knowledge-heavy tasks; post-compression
fine-tuning study (Sep 2026) — brief FT on small data recovers substantial perf,
cheaper than full KD.

Method: train the pruned model on the BASE model's own answers (self-distillation).
Routed experts stay FROZEN — they are stored as fused 3-D params the LoRA wrapper
cannot wrap. LoRA (rank ~64) is applied to attention, shared-expert, and other
projections; the router is trained fully. Loss masked to the assistant turn.

Relevance for us: the capability-restoring heal to run on real god RE tool-call
traces, so the pruned model re-learns to drive list/read/disasm/bash cleanly.
Combine with Router KD (router) — they touch different params.

Related: `router-kd.md`, `moe-pruner-kd.md`.
