# bits_per_weight.py — effective bits per weight of a converted checkpoint (packed ints + scales + biases).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
# Old version assumed 4 bits for every quantized tensor and counted packed uint32 words as params,
# so mixed 2/3/6-bit recipes logged wrong bpw. Now reads each prefix's bits from quant_cfg.


def bits_per_weight(out: dict, quant_cfg: dict) -> float:
    total_bits = 0
    total_params = 0
    for k, v in out.items():
        if not k.endswith(".weight"):
            continue
        prefix = k[: -len(".weight")]
        if prefix + ".scales" in out:                       # quantized affine triple
            bits = int(quant_cfg.get(prefix, {}).get("bits", quant_cfg["bits"]))  # per-prefix or default
            total_params += v.size * 32 // bits             # each uint32 packs 32/bits weights
            total_bits += v.nbytes * 8 + out[prefix + ".scales"].nbytes * 8
            total_bits += out[prefix + ".biases"].nbytes * 8 if prefix + ".biases" in out else 0
        else:
            total_params += v.size
            total_bits += v.nbytes * 8                      # float tensor: dtype width
    return total_bits / max(total_params, 1)
