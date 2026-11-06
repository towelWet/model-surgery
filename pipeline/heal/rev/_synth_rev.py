# _synth_rev.py — generate one synthetic reverse-engineering training row.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
from _chat_row import _row                   # heal/code on path (build script)
from _balanced_data import (
    ASMS, CTX, FIELDS, FMTS, FUNCS, INSNS,
    REV_BINARY, REV_DISASM, REV_MAL, REV_VULN,
)


def _synth_rev(rng):
    """Build one synthetic RE training row from random templates."""
    pool = REV_DISASM + REV_BINARY + REV_VULN + REV_MAL
    tpl, stub, tag = rng.choice(pool)
    user = tpl.format(
        asm=rng.choice(ASMS), insn=rng.choice(INSNS),
        ctx=rng.choice(CTX), fmt=rng.choice(FMTS),
        field=rng.choice(FIELDS), off=rng.randint(0x10, 0x400),
        fmt_a=rng.choice(FMTS), fmt_b=rng.choice(FMTS),
        fn=rng.choice(FUNCS),
        code="char buf[8]; gets(buf);",
        hint=rng.choice(["ROP", "heap overflow", "format string", "UAF"]),
        stage=rng.choice(["loader", "C2 beacon", "credential dumper"]),
        blob="0a 01 00 00 04 00 00 00",
    )
    asst = f"{stub}: expert RE answer — identify structure, risk, and next analysis step."
    return _row(user, asst, tag)
