# Code/reversing heal

Build the local corpus from the skill directory:

```sh
python3 pipeline/heal/build/code_rev_corpus.py --rows 5000
python3 pipeline/heal/tool_distill.py --student "$SURGERY_STUDENT_MODEL" \
  --data pipeline/heal/data/code_rev_corpus.jsonl \
  --teacher-cache "$SURGERY_TEACHER_CACHE" --adapter-path "$SURGERY_ADAPTER_PATH"
```

The trainer is tensor-array-train. Historical quantized checkpoints encountered a packed
input-width mismatch; a compatible trainer and a successful run must be proven before claiming
healing. A banned mlx_lm fallback is not provided. No training is started by loading this skill.

The current tool/code/reversing suites are diagnostic-only. Their source fingerprints make
comparisons inspectable; they do not establish executable capability retention.
