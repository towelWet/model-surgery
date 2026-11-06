# docs.py — doctor check: script paths named in skill docs (`profile/x.py`, `eval/tool/run.py` …) exist.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
# Resolves each backticked path against: skill root, pipeline/, and the doc's own folder.
# Skips third-party Reference/ clones and db/rejected/ (quarantine records describe old paths on purpose).
import os
import pathlib
import re

REF = re.compile(r"`([\w./~-]+\.(?:py|sh|command|md|toml))[^`]*`")  # first token of a backticked command/path
ABS = re.compile(r"(?<![\w/])(/Users/[^/\s]+/\.claude/skills/model-surgery/[^\s`|)]+)")  # absolute skill paths
SKIP = ("Reference/", "db/rejected/", "research/", "db/qwen/")  # db/<family>: dated run logs keep paths as run


def run(root):
    skill = pathlib.Path(root).parent
    out = []
    for doc in sorted(list(skill.rglob("*.md")) + [skill / "SKILL.md"]):
        rel = str(doc.relative_to(skill))
        if rel.startswith(SKIP) or ".grit" in doc.parts:
            continue
        for n, line in enumerate(doc.read_text(errors="ignore").splitlines(), 1):
            for ab in ABS.findall(line):
                if not pathlib.Path(ab.rstrip(".,:;")).exists():
                    out.append(f"{rel}:{n} {ab}")
            for ref in REF.findall(line):
                if ref.startswith(("~", "/")) or "<" in ref or "github.com" in ref:
                    continue                                # absolute handled above; placeholders / remote repos skipped
                bases = (skill, skill / "pipeline", doc.parent,
                         *(skill / "pipeline").iterdir())  # stage-relative names in per-stage tables
                if not any((b / ref).exists() for b in bases):
                    out.append(f"{rel}:{n} {ref}")
    return sorted(set(out))


CHECK = {"name": "docs", "help": "script paths cited in docs exist", "run": run}
