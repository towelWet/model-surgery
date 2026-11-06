# fig_stack.py — Figure 1: the 64-layer parent beside its 48-layer pruned child.
# Why: depth pruning is the easiest family to see; the reader needs one picture
# showing which layers were removed and that both ends survived untouched.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.

from theme import plt, save, BLUE, ORANGE, FAINT, MUTED, INK, RULE

CUTS = set(range(4, 16)) | set(range(40, 44))               # the winning spec "4-15,40-43" from profiling


def draw(path):
    fig, ax = plt.subplots(figsize=(6.4, 3.0))              # wide-short: two stacks read side by side
    for i in range(64):                                     # parent: one thin bar per layer, bottom-up
        cut = i in CUTS                                     # cut layers get the orange/hatched treatment
        ax.add_patch(plt.Rectangle((0, i), 0.8, 0.82,       # 0.82 of 1.0 leaves the 2px surface gap
                                   facecolor=ORANGE if cut else BLUE,
                                   edgecolor=RULE, linewidth=0.3,
                                   hatch="///" if cut else None))   # texture = secondary encoding, not colour alone
    kept = [i for i in range(64) if i not in CUTS]           # survivors, in original order
    for j, _ in enumerate(kept):                            # child: survivors restacked with no gaps
        ax.add_patch(plt.Rectangle((1.6, j), 0.8, 0.82, facecolor=BLUE,
                                   edgecolor=RULE, linewidth=0.3))
    ax.annotate("", xy=(1.55, 32), xytext=(0.9, 32),        # parent -> child arrow at mid-height
                arrowprops=dict(arrowstyle="-|>", color=MUTED, linewidth=1.2))
    ax.text(1.22, 34, "surgery", ha="center", color=MUTED, fontsize=9)
    ax.text(0.4, -2.5, "Parent\n64 layers", ha="center", va="top", color=INK, fontsize=9.5)
    ax.text(2.0, -2.5, "Child\n48 layers", ha="center", va="top", color=INK, fontsize=9.5)
    ax.text(-0.12, 9, "cut 4–15", color=ORANGE, fontsize=8.5, va="center", ha="right")   # cuts labelled left
    ax.text(-0.12, 41, "cut 40–43", color=ORANGE, fontsize=8.5, va="center", ha="right")
    ax.text(0.92, 1.5, "first 4 kept", color=MUTED, fontsize=8, ha="left")    # never-cut ends, right side
    ax.text(0.92, 61, "last 4 kept", color=MUTED, fontsize=8, ha="left")
    ax.set_xlim(-1.2, 2.8); ax.set_ylim(-13, 68)             # room for the labels outside the stacks
    ax.axis("off")                                          # a schematic has no axes to read
    return save(fig, path)
