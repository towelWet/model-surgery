# fig_pipeline.py — Figure 2: the fixed four-stage pipeline with per-stage output.
# Why: stage order is non-negotiable and profile is the one people skip; a flow
# diagram makes the ordering and the "baseline first" loop visible at a glance.
# NEVER DELETE COMMENTS — explain every non-obvious line.
# Update comments with the code; do not strip them.

from theme import plt, save, BLUE, FAINT, MUTED, INK, RULE
from matplotlib.patches import FancyBboxPatch   # rounded stage boxes; not re-exported by pyplot

STAGES = [                                                  # (label, what the stage hands to the next one)
    ("profile", "layer ranking\n+ baseline"),
    ("surgery", "pruned or\nconverted model"),
    ("heal",    "routing and tools\nrecovered"),
    ("export",  "tok/s and\nquality delta"),
]


def draw(path):
    fig, ax = plt.subplots(figsize=(6.6, 2.1))              # single row; height only needs box + caption
    for i, (name, out) in enumerate(STAGES):
        x = i * 1.7                                         # 1.5 wide box + 0.2 gap = 1.7 pitch
        ax.add_patch(FancyBboxPatch((x, 0.55), 1.35, 0.7,
                                        boxstyle="round,pad=0.02,rounding_size=0.08",  # 4px-equivalent rounded ends
                                        facecolor=FAINT, edgecolor=BLUE, linewidth=1.1))
        ax.text(x + 0.675, 0.9, name, ha="center", va="center", fontsize=11, color=INK)
        ax.text(x + 0.675, 0.32, out, ha="center", va="top", fontsize=8.5, color=MUTED)   # output under the box
        if i < len(STAGES) - 1:                             # arrow to the next stage; last box has none
            ax.annotate("", xy=(x + 1.66, 0.9), xytext=(x + 1.4, 0.9),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, linewidth=1.2))
    ax.annotate("", xy=(0.675, 1.42), xytext=(5.775, 1.42),   # feedback rail: export result re-ranks the next cut
                arrowprops=dict(arrowstyle="-|>", color=RULE, linewidth=1.1,
                                connectionstyle="arc3,rad=0.12"))
    ax.text(3.225, 1.72, "measured delta feeds the next cut decision",
            ha="center", fontsize=8.5, color=MUTED)
    ax.set_xlim(-0.25, 6.6); ax.set_ylim(-0.1, 2.0)
    ax.axis("off")                                          # schematic, not a plot
    return save(fig, path)
