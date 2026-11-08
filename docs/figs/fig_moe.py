# fig_moe.py — Figure 3: one dense FFN partitioned into experts plus a router.
# Why: "dense to MoE" sounds like new weights are invented; the picture shows the
# parent's 17408 neurons being split, with only the small router added.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.

from theme import plt, save, BLUE, ORANGE, FAINT, MUTED, INK, RULE

N_SHOWN = 8                                                 # 8 drawn shards stand in for the real 64 experts


def draw(path):
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    ax.add_patch(plt.Rectangle((0, 0.3), 1.15, 1.6, facecolor=FAINT,     # the parent FFN as one solid block
                               edgecolor=BLUE, linewidth=1.1))
    ax.text(0.575, 1.1, "dense FFN\n17,408", ha="center", va="center", fontsize=9.5, color=INK)
    ax.annotate("", xy=(1.95, 1.1), xytext=(1.25, 1.1),     # the partition step
                arrowprops=dict(arrowstyle="-|>", color=MUTED, linewidth=1.2))
    ax.text(1.6, 1.28, "partition", ha="center", fontsize=8.5, color=MUTED)
    for i in range(N_SHOWN):                                # experts: equal shards, 2px gap between fills
        y = 1.82 - i * 0.2                                  # stacked downward from the top of the block
        ax.add_patch(plt.Rectangle((2.05, y), 0.85, 0.16, facecolor=BLUE,
                                   edgecolor=RULE, linewidth=0.3))
    ax.text(2.475, 2.12, "64 experts × 192 neurons", ha="center", fontsize=9, color=INK)
    ax.text(2.475, 0.05, "every neuron lands in exactly one expert",
            ha="center", fontsize=8.5, color=MUTED)         # the invariant that makes this lossless
    ax.add_patch(plt.Rectangle((3.35, 0.95), 0.75, 0.32, facecolor="none",  # the only new tensor
                               edgecolor=ORANGE, linewidth=1.3, hatch="///"))
    ax.text(3.725, 1.11, "router", ha="center", va="center", fontsize=9, color=INK,
            bbox=dict(facecolor="#fcfcfb", edgecolor="none", pad=1.5))   # knock the hatch out behind the label
    ax.text(3.725, 0.72, "the only\nnew weights", ha="center", va="top", fontsize=8.5, color=ORANGE)
    ax.annotate("", xy=(3.3, 1.11), xytext=(2.95, 1.11),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, linewidth=1.2))
    ax.text(2.25, -0.35, "64 × 192  +  5,120 shared  =  17,408   (exact — nothing invented)",
            fontsize=9, color=INK, ha="center")   # the arithmetic that proves nothing was invented
    ax.set_xlim(-0.2, 4.7); ax.set_ylim(-0.55, 2.4)
    ax.axis("off")
    return save(fig, path)
