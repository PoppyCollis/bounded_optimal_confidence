"""Two-part figure: the dot stimulus (A) and the four stimulus configs' likelihoods (B).

    A: config 1 as the subject sees it. 375 dots per category, drawn from isotropic 2D Gaussians centred on
       (mu, 0) with sigma 64, in the shared state colours. One of the 1125 dots is the target: black, a little
       larger, with a white ring, placed at TARGET and labelled.
    B: 2x2, one panel per config, the likelihoods p(s|x) as 1D Gaussians, sigma 64, s in [-300, 300].

    1: mu = (-96, 0, 96)    2: mu = (-128, 0, 128)
    3: mu = (-96, -59, 96)  4: mu = (-96, 59, 96)

Solid lines in the shared state colours (left blue, middle orange, right green), no titles, no ticks in B (axis labels on its
bottom-left panel only), bold config number under each panel and a bold letter under each part. B's panels share a y-axis so they compare
directly. Text sizes, line width and axis styling are the shared ones (`plotting/style.py`), so this figure matches the
others.

Layout: A's x-axis is level with the x-axes of B's bottom row and A's y-axis runs up to the top of B's top row, so
A is a square as tall as the whole of B. A's legend sits above that, higher than anything in B.

The configs and sigma are those of `experiment.CATEGORY_X` / `SIG_S`, written out here as the figure script had them.
From `ITBR_confidence/paper_plots/fig_config_likelihoods.py`, unchanged but for the imports.

The output depends very slightly on Python's hash seed (the old script's too): in about one run in four the
figure's tight bounding box comes out a rounding step different, which moves the Target arrow by a fraction
of a pixel (79 pixels of the PNG). `PYTHONHASHSEED=0` gives the committed version.

Run: python paper_plots/fig_config_likelihoods.py
    -> paper_plots/figures/fig_config_likelihoods.png (and .svg)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

from bounded_optimal_confidence.plotting.readout_traces import likelihoods
from bounded_optimal_confidence.plotting.style import (CATEGORY_LABELS, LABEL_SIZE, LEGEND_SIZE, LINE_WIDTH,
                                                       STATE_COLORS, TICK_SIZE, TITLE_SIZE, use_paper_style)

FIG_DIR = Path(__file__).resolve().parent / "figures"

CONFIGS = {
    1: (-96.0, 0.0, 96.0),
    2: (-128.0, 0.0, 128.0),
    3: (-96.0, -59.0, 96.0),
    4: (-96.0, 59.0, 96.0),
}
SIGMA = 64.0
W = np.linspace(-300, 300, 1501)
# One factor on every text size (and the number's offset), so they grow in proportion.
# 1.0 gives the same sizes as the other figures.
TEXT_SCALE = 1.6
FIGSIZE = (21, 9)   # A's legend is added above this when the image is saved
LETTER_OFFSET = -92 * TEXT_SCALE   # part letters, in points below the x-axes

# Panel A
DOTS_CFG = 1
N_DOTS = 375          # per category
SEED = 2              # all 1125 dots land inside the axes (seeds 0 and 1 each put one dot past |s_x| = 300)
UY = 0.0              # every category is centred on s_y = 0
LIM = 300             # both axes run over [-LIM, LIM]
DOT_TICKS = [-LIM, 0, LIM]   # the outer ticks sit on the ends of the axes
DOT_SIZE = 26
TARGET = (50.0, 50.0)          # the sampled dot closest to this point becomes the target and is moved onto it
TARGET_SIZE = 135              # marker area, against DOT_SIZE for the rest
TARGET_RING = 2.5              # width of the white ring round the target, in points
TARGET_TEXT = (185.0, 220.0)   # where the word sits; the arrow runs from here to the target


def draw_dots(ax):
    """Panel A: the sampled dots of config DOTS_CFG, one of them black and labelled as the target."""
    rng = np.random.default_rng(SEED)
    mu = np.array(CONFIGS[DOTS_CFG])
    centres = np.column_stack([mu, np.full(len(mu), UY)])
    pts = (centres[:, None, :] + SIGMA * rng.standard_normal((len(mu), N_DOTS, 2))).reshape(-1, 2)
    cat = np.repeat(np.arange(len(mu)), N_DOTS)
    target = np.argmin(np.sum((pts - np.array(TARGET)) ** 2, axis=1))
    sampled = pts[target].copy()
    pts[target] = TARGET

    # mixed draw order, so no category sits on top of the others
    order = rng.permutation(len(pts))
    order = order[order != target]
    ax.scatter(pts[order, 0], pts[order, 1], c=[STATE_COLORS[k] for k in cat[order]], s=DOT_SIZE, linewidths=0)
    # the ring is a white disc under the black one (marker area is diameter squared, in points)
    ring_diameter = np.sqrt(TARGET_SIZE) + 2 * TARGET_RING
    ax.scatter(*pts[target], color="white", s=ring_diameter ** 2, linewidths=0, zorder=3)
    ax.scatter(*pts[target], color="black", s=TARGET_SIZE, linewidths=0, zorder=4)
    ax.annotate("Target", xy=pts[target], xytext=TARGET_TEXT, ha="center", va="bottom", color="black",
                fontsize=LEGEND_SIZE * TEXT_SCALE,
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5, shrinkA=2, shrinkB=ring_diameter / 2 + 2))

    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal")
    ax.set_xticks(DOT_TICKS)
    ax.set_yticks(DOT_TICKS)
    # extra pad, so the two corner labels (the lowest tick on each axis) do not touch
    ax.tick_params(labelsize=TICK_SIZE * TEXT_SCALE, pad=0.4 * TICK_SIZE * TEXT_SCALE)
    ax.set_xlabel("$s_x$", fontsize=LABEL_SIZE * TEXT_SCALE)
    ax.set_ylabel("$s_y$", fontsize=LABEL_SIZE * TEXT_SCALE)
    ax.spines[["top", "right"]].set_visible(False)

    handles = [Line2D([], [], color=c, marker="o", ls="none", markersize=9 * TEXT_SCALE, label=name)
               for name, c in CATEGORY_LABELS.items()]
    # A figure legend anchored to the axes: the layout ignores it, so it sits above the panel without pushing the
    # axes down, and the saved image still grows to hold it.
    ax.figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02), bbox_transform=ax.transAxes,
                     ncol=len(handles), fontsize=LEGEND_SIZE * TEXT_SCALE, handletextpad=0.1, columnspacing=0.8,
                     borderaxespad=0)

    outside = np.sum(np.any(np.abs(pts) > LIM, axis=1))
    print(f"cfg {DOTS_CFG} dots: {len(pts)} drawn, {outside} outside [-{LIM}, {LIM}]^2; "
          f"target at ({pts[target, 0]:.1f}, {pts[target, 1]:.1f}), a category {cat[target] + 1} dot "
          f"sampled at ({sampled[0]:.1f}, {sampled[1]:.1f})")


def draw_likelihood_panels(axes):
    """Panel B: one axis per config, its three likelihoods."""
    for ax, (cfg, mu) in zip(axes.flat, CONFIGS.items()):
        lik = likelihoods(W, mu, SIGMA)
        for k in range(len(mu)):
            ax.plot(W, lik[:, k], color=STATE_COLORS[k], lw=LINE_WIDTH)
        ax.set_xlim(W[0], W[-1])
        ax.set_xticks([])
        ax.set_ylim(0, None)
        ax.set_yticks([])
        ax.spines[["top", "right"]].set_visible(False)
        # config number, centred under the panel, below where the x-axis label goes
        ax.annotate(str(cfg), xy=(0.5, 0), xycoords="axes fraction", xytext=(0, -52 * TEXT_SCALE), textcoords="offset points",
                    fontsize=TITLE_SIZE * TEXT_SCALE, fontweight="bold", ha="center", va="top")
        print(f"cfg {cfg} mass inside [-300, 300]: {np.round(np.trapezoid(lik, W, axis=0), 4)}")

    # axis labels on the bottom-left panel only; they hold for all four
    axes[1, 0].set_xlabel("$s$", fontsize=LABEL_SIZE * TEXT_SCALE)
    axes[1, 0].set_ylabel("$p(s\\,|\\,x)$", fontsize=LABEL_SIZE * TEXT_SCALE)


def square_up(fig, grid, ax):
    """Set the left column's width so panel A's slot is square. A's axes are square, so they then fill the slot:
    bottom on the x-axes of B's bottom row, top on the top of B's top row."""
    ratios = list(grid.get_width_ratios())
    for _ in range(20):
        fig.draw_without_rendering()
        slot = ax.get_position(original=True)
        width, height = slot.width * fig.get_figwidth(), slot.height * fig.get_figheight()
        if abs(width - height) < 1e-3:
            return
        ratios[0] *= height / width
        grid.set_width_ratios(ratios)
    raise RuntimeError(f"panel A's slot did not come out square: {width:.3f} x {height:.3f} in")


def main():
    use_paper_style()
    fig = plt.figure(figsize=FIGSIZE, layout="constrained")
    grid = fig.add_gridspec(2, 3, width_ratios=[1.4, 1, 1])
    ax_dots = fig.add_subplot(grid[:, 0])
    axes = np.empty((2, 2), dtype=object)
    for r in range(2):
        for c in range(2):
            axes[r, c] = fig.add_subplot(grid[r, c + 1], sharey=axes[0, 0])

    draw_dots(ax_dots)
    draw_likelihood_panels(axes)

    # part letters, the same distance below the (level) x-axes: A under its panel, B under the middle of the 2x2
    letter_kw = dict(xycoords="axes fraction", textcoords="offset points", fontsize=TITLE_SIZE * TEXT_SCALE,
                     fontweight="bold", ha="center", va="top")
    ax_dots.annotate("A", xy=(0.5, 0), xytext=(0, LETTER_OFFSET), **letter_kw)
    letter_b = axes[1, 0].annotate("B", xy=(1, 0), xytext=(0, LETTER_OFFSET), **letter_kw)
    letter_b.set_in_layout(False)   # moved after the layout is settled, so it must not feed back into it

    square_up(fig, grid, ax_dots)
    # centre B on the gap between its two columns
    gap = (axes[1, 1].get_position().x0 - axes[1, 0].get_position().x1) * fig.get_figwidth() * 72
    letter_b.xyann = (gap / 2, LETTER_OFFSET)

    edges = {name: np.round(ax.get_position().get_points().ravel() * np.tile(fig.get_size_inches(), 2), 3)
             for name, ax in (("A", ax_dots), ("B top-left", axes[0, 0]), ("B bottom-left", axes[1, 0]))}
    print("axes [x0 y0 x1 y1] in inches: " + "; ".join(f"{k} {v}" for k, v in edges.items()))

    FIG_DIR.mkdir(exist_ok=True)
    for suffix in (".png", ".svg"):   # SVG is vector; dpi only affects any rasterised parts
        out = FIG_DIR / f"fig_config_likelihoods{suffix}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"saved {out}")


if __name__ == "__main__":
    main()
