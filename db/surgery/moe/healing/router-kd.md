# Router KD — router-only knowledge distillation (post-compression heal)

Source: "Is Retraining-Free Enough? The Necessity of Router Calibration for
Efficient MoE Compression", Hyeon & Do, arXiv 2603.02217 (2026).

Problem it names: after retraining-free compression (prune / edit / merge), the
leftover degradation is mostly a ROUTER↔EXPERT MISMATCH — the experts changed but
the router still routes as if they hadn't. On fine-grained MoE this shows up as the
model emitting the wrong expert mix, which downstream reads as broken behaviour.

Method: train ONLY the router (a tiny fraction of params; experts frozen) to match
the ORIGINAL uncompressed model's next-token distribution on unlabeled calibration
text. No labels, no expert retraining.

Results: consistent recovery across all three paradigms (pruning, editing, merging);
gains are SUBSTANTIALLY larger on fine-grained MoE (many small experts) than coarse
MoE, attributed to more complex routing boundaries.

Relevance for us (CRITICAL): this is the heal we never ran. Our pruned variants
(REAP/AIMER t61-t72) emitted empty/garbage TOOL ARGS — a routing failure, exactly
the mismatch Router KD targets. Qwen3-30B-A3B is fine-grained (128 experts) = the
best case. Apply after ANY prune, before trialing tool-calling.

Learning rate hint (not Router KD specific): @VukRosic99 2026-10-22 found MoE router
+ norm gains fine at 1x base AdamW LR while embeddings wanted 8-32x. Start router KD at
base LR; do not scale it up first.

Related: `../pruning/hope.md`, `moe-pruner-kd.md`, `../scoring/reap.md`.
