# Eval (`pipeline/eval/`)

| Suite | Cases | Metric |
|---|---|---|
| `tool/run.py` | 104 | tool selection accuracy + argument validity |
| `code/run.py` | 17 | Python / frontend / review graded by required tokens |
| `rev/run.py` | 16 | x86 / ARM64 / formats / exploitation / decompiler |
| `kl/run.py` | ~30 prompts | mean KL(parent ‖ student) on general text |

Cases live in `<suite>/cases/<category>.py` (one `CASES` list per file, hot-loaded).
Loaders reject duplicate cases/IDs, empty selections and impossible tool expectations.
Reports preserve numeric precision and include an evaluation-source fingerprint.
Token grader: every `need` item must appear; an item that is a list accepts any one spelling
(`["jnz", "jne"]`), so correct paraphrases pass. Prompts never name the answer. Tool cases may
list several acceptable tools (read: `read_file` or `bash`); argument validity counts every
case that needs a tool, so a missed call counts against it.
Current graders are diagnostic: token matching does not prove code execution or reversing,
and KL compares renormalized top-k heads. Reports declare `diagnostic_only = true`.
The promotion gate rejects these reports, legacy reports without evidence metadata,
changed fingerprints/counts, non-finite scores, and empty/report-only rules.
There is currently no promotion-grade eval path. See [CLI details](docs/cli.md).
