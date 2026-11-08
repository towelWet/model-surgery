<!-- expert-prefetch.md — stream MoE experts from SSD with prediction/overlap; prior art next to our SLAB. -->
<!-- NEVER DELETE COMMENTS — explain every non-obvious line. Update comments with the code; do not strip them. -->

# Expert streaming + prefetch — outside prior art

All from X reposts; claims are the posters', not verified by us.

| Project | Claim | Source |
|---|---|---|
| Edge0 | 35B MoE on 24 GB Mac mini, 2.9 GB peak; prediction head guesses next step's experts, SSD read overlaps compute, decode +59%; M4 Pro 15–18 tok/s; quality 79.2 vs fp16 83.2 | @Ryrenz 2026-10-29, github.com/Edge0-AI/Edge0 |
| Colibri | GLM-5.2 744B on 25 GB RAM; dense part resident, experts streamed from disk; one 2,400-line C file, CPU only | @chenzeling4 2026-07-11, github.com/JustVugg/colibri |
| Project Maya | each expert sent to GPU as it arrives from SSD; GLM decode +22% | @ItsmeAjayKV 2026-10-30 |
| MiniCPM-V 4.7 | 35B-A3B multimodal on phone in ~3 GB; LES = Lookahead Expert Scheduling | @OpenBMB 2026-10-13, github.com/OpenBMB |
| Kapitsa | Qwen3.6-35B-A3B 4-bit 154 tok/s on M4 Pro 48 GB (llama.cpp+MTP 69); Metal kernel tuning + DFlash2 spec decode | @Oluwaphilemon1 2026-10-24 |

## Relevance for us

**Calibration blocker.** Our calibration / KD forward passes hit the Metal GPU watchdog
while streaming experts (see `project-tau-35b-a3b-surgery-campaign` memory). Colibri's
CPU-only streamed forward and Edge0's read/compute overlap are two outside designs for
exactly that pass.

**Speed.** Our SLAB served intact at 3.3 tok/s in steered trial intact/2-test/1
(2026-11-02, ECAP 32); Kapitsa and Edge0 report 15–154 tok/s on similar Macs. Gap is large enough that trial
turns (1582 s on 2026-11-02) are engine-bound, not model-bound.

Related: `../INDEX.toml` [serving] slab.
