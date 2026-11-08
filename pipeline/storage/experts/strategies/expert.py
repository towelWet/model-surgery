# expert.py — split strategy "expert" (default): shared file + one file per G routed experts (all layers).
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.
# Hot-loadable: any strategies/*.py with a STRATEGY dict is discovered; no registry edits.


def file_of(info, group, lay):
    """Routed expert e -> expert_file[e // G]; everything else -> shared_file."""
    if info["expert"] is None:
        return lay["shared_file"]                                  # attention, norms, router, shared expert
    return lay["expert_file"].format(index=info["expert"] // group)


STRATEGY = {"name": "expert", "help": "shared + experts-NNN (G experts per file); selective expert loading",
            "run": file_of}                                        # run = file_of (plugin contract key)
