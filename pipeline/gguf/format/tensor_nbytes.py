# tensor_nbytes.py — payload size of one ggml tensor from its dims and type.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
import math

from .tables import QUANT_SIZES


def tensor_nbytes(dims, ggml_type):
    if ggml_type not in QUANT_SIZES:
        raise ValueError(f"unsupported ggml type {ggml_type}")
    block, size = QUANT_SIZES[ggml_type]
    if dims[0] % block:                                     # rows are whole blocks along ne0
        raise ValueError(f"ne0 {dims[0]} not a multiple of block {block}")
    return math.prod(dims) // block * size
