# balanced/_data.py — tags and synthetic rev templates (data only).
REV_TAGS = frozenset({"disasm", "binary", "review"})
CODE_TAGS = frozenset({"dom", "frontend", "testing", "clean code", "stdlib patterns"})
SYSTEM = "You are an expert reverse-engineering and systems security assistant."

REV_DISASM = [
    ("Analyze this x86_64 snippet for control flow:\n{asm}", "cf", "disasm"),
    ("What does ARM64 instruction `{insn}` do in context of `{ctx}`?", "arm", "disasm"),
]
REV_BINARY = [
    ("Parse {fmt} {field} at offset 0x{off:x}; what fails if truncated?", "hdr", "binary"),
    ("Compare {fmt_a} vs {fmt_b} load commands for PIE binaries.", "lc", "binary"),
]
REV_VULN = [
    ("Spot the bug in this decompiled `{fn}`:\n{code}", "vuln", "review"),
    ("Classify exploit primitive: {hint}", "prim", "review"),
]
REV_MAL = [
    ("Malware stage: {stage}. IOCs and next pivot?", "mal", "review"),
    ("Decode this protocol field layout: {blob}", "proto", "review"),
]
FMTS = ["ELF64", "Mach-O 64", "PE32+"]
FIELDS = ["e_phoff", "mach_header", "IMAGE_OPTIONAL_HEADER"]
FUNCS = ["parse_tlv", "read_u32_le", "decrypt_cbc"]
ASMS = [
    "48 8b 05 ?? ?? ?? ??\n48 85 c0\n74 0a\nff e0",
    "55 48 89 e5\n48 8d 3d 00 00 00 00\ne8 ?? ?? ?? ??",
]
INSNS = ["ldr x0, [x8]", "bl _malloc", "cbz w0, #0x40"]
CTX = ["objc_msgSend stub", "fork+exec chain", "JNI RegisterNatives"]
