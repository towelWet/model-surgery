# formats.py — ELF/Mach-O/PE structure; data only.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.

CASES = [
    {
        "id": "elf_magic",
        "prompt": "First 4 bytes of ELF and meaning?",
        "need": [["0x7f", "7f 45 4c 46"], "elf"],          # hex-dump spelling is equally right
        "max_tokens": 140,
    },
    {
        "id": "macho_magic",
        "prompt": "Mach-O 64-bit little-endian magic constant (hex)?",
        "need": [["feedfacf", "cf fa ed fe"]],             # constant or its little-endian bytes
        "any_of": ["0xfeedfacf", "feedfacf"],
        "max_tokens": 120,
    },
    {
        "id": "pe_dos",
        "prompt": "PE file starts with DOS stub; which offset field points to PE signature?",
        "need": ["0x3c", "e_lfanew"],
        "max_tokens": 180,
    },
    {
        "id": "elf_got",
        "prompt": "ELF: what is .got.plt used for at runtime?",
        "need": ["got", "plt"],
        "any_of": ["lazy", "reloc", "dynamic"],
        "max_tokens": 180,
    },
    {
        "id": "pe_import",
        "prompt": "PE32+: where does the loader resolve IAT entries?",
        "need": ["import"],
        "any_of": ["iat", "thunk", "dll"],
        "max_tokens": 180,
    },
]
