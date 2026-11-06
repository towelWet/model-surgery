# Active-B decode ceilings (companion to SKILL_TECHNIQUES_PRUNE.md)

`bytes/tok = active_B × bits/8`. `tok/s_ceil ≈ BW / bytes/tok`.

From historical local hardware notes (~85 GB/s, 32 GB Apple Silicon — not the 120 GB/s Bonsai row):

| Active | Quant | GB/tok | Ceiling |
|---|---|---|---|
| 27B dense | 3-bit | 10.1 | ~8 tok/s |
| 20.8B (48L) | 3-bit | 7.8 | ~11 tok/s |
| 17.8B Whittle top-16 | 3-bit | 6.7 | ~13 tok/s |
| ~5B top-4 MoE | 3-bit | 1.9 | ~45 tok/s |
| ~3B top-4 + depth | 3-bit | 1.1 | ~77 tok/s |

Bonsai-27B-mlx-1bit (overnight estimate): **120 GB/s / ~4.2 GB ≈ ~28 tok/s**.
README native **3.9 GB**; MLX pack **5.13 GB** on disk. Not loaded tonight.
Morning gate >21.5 tok/s: MORNING_20260921.md [missing: not on disk 2026-11-06].
