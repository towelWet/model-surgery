# _genome_merge.py — run one merge from genome with fixed method (TIES, DARE, task arithmetic, ...).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
#
# Genome scalars are one-element lists (uniform shape for _mutate). Last run_merge arg False =
# keep source format (bf16 stays bf16, quantized sources are requantized), same as soup.
from pipeline.surgery.merge._ties import run_merge


def genome_merge(base, finetunes, method, genome, out, seed):
    spec = {"method": method, "density": genome["density"][0], "lambda": genome["lambda"][0],
            "weights": list(genome["weights"])}
    run_merge([base, *finetunes], out, spec, seed, False)
