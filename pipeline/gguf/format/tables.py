# tables.py — GGUF constants: value-type codes and ggml tensor block sizes (data only).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
# Source of truth: llama.cpp gguf-py/gguf/constants.py (GGUFValueType, GGML_QUANT_SIZES).

MAGIC = 0x46554747                                        # "GGUF" little-endian
DEFAULT_ALIGNMENT = 32                                    # used when general.alignment is absent
STRING, ARRAY = 8, 9                                      # value types needing custom encoding
SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
           10: "<Q", 11: "<q", 12: "<d"}                  # struct formats per GGUFValueType
QK_K = 256                                                # k-quant super-block width
QUANT_SIZES = {                                           # ggml type id -> (elements per block, bytes per block)
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34), 9: (32, 40),
    10: (256, 2 + 2 + QK_K // 16 + QK_K // 4), 11: (256, 2 + QK_K // 4 + QK_K // 8 + 12),
    12: (256, 2 + 2 + QK_K // 2 + 12), 13: (256, 2 + 2 + QK_K // 2 + QK_K // 8 + 12),
    14: (256, 2 + QK_K // 2 + QK_K // 4 + QK_K // 16), 15: (256, 4 + QK_K + QK_K // 8),
    16: (256, 2 + QK_K // 4), 17: (256, 2 + QK_K // 4 + QK_K // 32), 18: (256, 2 + QK_K // 4 + QK_K // 8),
    19: (256, 2 + QK_K // 8 + QK_K // 16), 20: (32, 18), 21: (256, 2 + QK_K // 4 + QK_K // 8 + QK_K // 32 + 4),
    22: (256, 2 + QK_K // 4 + QK_K // 16), 23: (256, 2 + 2 + QK_K // 2 + QK_K // 64),
    24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8),
    29: (256, QK_K // 8 + QK_K // 16 + QK_K // 32), 30: (1, 2), 34: (256, 2 + 4 * 13), 35: (256, 2 + 64),
    39: (32, 17),
}
