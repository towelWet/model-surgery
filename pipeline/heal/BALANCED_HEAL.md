# Balanced code/reversing heal

The old notes used a banned mlx_lm fallback. That route is quarantined and is not an
instruction to run it. Use tensor-array-train; unsupported packed checkpoints remain blocked.

From the skill directory:

```sh
python3 pipeline/heal/build/rev_balanced_corpus.py --rows 3000 --rev-ratio 0.6
python3 pipeline/heal/tool/distill.py --student "$SURGERY_STUDENT_MODEL" \
  --data pipeline/heal/data/rev_balanced_corpus.jsonl \
  --teacher-cache "$SURGERY_TEACHER_CACHE" --adapter-path "$SURGERY_ADAPTER_PATH"
```

These commands are examples, not a verified successful training run. Check memory headroom
and trainer compatibility before execution. Never run teacher and student resident together.
Current eval scores are diagnostic-only and cannot authorize promotion.
See `pipeline/heal/CODE_REV_TRAIN.md` for the same trainer's code/reversing workflow.
