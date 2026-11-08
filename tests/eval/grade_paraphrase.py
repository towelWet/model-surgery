"""Grader accepts equivalent spellings and x86/ELF/two_sum cases pass correct paraphrases (QOL chat findings 2026-11-08)."""
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from pipeline.eval._grade import grade_response
from pipeline.eval.rev.cases.x86 import CASES as X86
from pipeline.eval.rev.cases.formats import CASES as FMT
from pipeline.eval.code.cases.python import CASES as PY


def test_paraphrases_pass_and_echo_fails():
    need = {c["id"]: c["need"] for c in X86 + FMT + PY}             # id -> need list
    jnz = need["x86_jnz"]
    assert grade_response("jne rel8: taken when ZF=0", jnz)          # no word "jump", still right
    assert not grade_response("jnz", jnz)                             # mnemonic alone, no condition
    assert grade_response("7F 45 4C 46 = \\x7fELF magic", need["elf_magic"])
    assert grade_response("def two_sum(n, t):\n seen = {}\n return []", need["py_two_sum"])
    assert "(mov + call)" not in next(c for c in X86 if c["id"] == "x86_call_rax")["prompt"]
