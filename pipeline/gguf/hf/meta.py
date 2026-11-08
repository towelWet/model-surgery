# meta.py — scalar GGUF metadata (ints, floats, strings) as plain Python dict via gguf-py reader.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.


def gguf_meta(reader):
    import gguf                                               # on sys.path via llamacpp_paths()
    out = {}
    for name, f in reader.fields.items():
        if len(f.data) != 1 or f.types[0] == gguf.GGUFValueType.ARRAY:
            continue                                          # arrays (tokens, merges) not needed here
        v = f.parts[f.data[0]]
        out[name] = bytes(v).decode("utf-8") if f.types[0] == gguf.GGUFValueType.STRING else v[0].item()
    return out
