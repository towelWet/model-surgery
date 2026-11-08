# x86.py — x86-64 decode + calling convention; data only.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.

CASES = [
    {
        "id": "x86_call_rax",
        "prompt": "x86-64 bytes 48 89 c8 then ff d0: give AT&T mnemonics for both.",
        "need": ["mov", "call"],
        "max_tokens": 180,
    },
    {
        "id": "x86_jnz",
        "prompt": "x86-64 opcode bytes 75 0a: mnemonic and when branch is taken.",   # no mnemonic hint: echo must not pass
        "need": [["jnz", "jne"], ["zf", "zero flag"]],          # condition is the meaning; word "jump" is not required
        "max_tokens": 160,
    },
    {
        "id": "x86_lea",
        "prompt": "Explain `48 8d 05` RIP-relative LEA in one sentence.",
        "need": ["lea", "rip"],
        "max_tokens": 160,
    },
    {
        "id": "x86_syscall",
        "prompt": "Linux x86-64: which register holds syscall number before `syscall` insn?",
        "need": ["rax"],
        "max_tokens": 120,
    },
]
