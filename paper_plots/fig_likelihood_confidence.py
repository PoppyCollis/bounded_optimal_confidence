"""2x2 panel: likelihoods (left column) and confidence readouts (right column), 2AFC (top) vs 3AFC (bottom).

    A: 2AFC likelihoods p(y|x), mu = (-1, 1), sigma = 1
    B: 2AFC confidence: Entropy, Max, Difference
    C: 3AFC likelihoods p(y|x), mu = (-2, 0, 2), sigma = 1.5
    D: 3AFC confidence: Entropy, Max, Difference, Entropy Top2

Styled to match the other paper figures: no box border (left and bottom axes only), large text, sparse ticks
and the shared palette. Likelihoods use the state colours of the main figure (blue, orange, green), all solid.
Readouts use colours outside that set (purple, aqua, coral, dark grey), so a readout never shares a colour with a
likelihood and each keeps its colour across B and D. Readouts are solid too, except where one lies exactly on
another (Max on Difference with two options): the later one is dashed so both colours show.

From `ITBR_confidence/paper_plots/fig_likelihood_confidence.py`: the drawing is unchanged; the readout helpers it
took from `bayes_1d_readouts` are in `plotting/readout_traces.py`, and the palette, sizes and rcParams it declared
are those of `plotting/style.py`.

Run: python paper_plots/fig_likelihood_confidence.py [--norm data|theory]
    -> paper_plots/figures/fig_likelihood_confidence.png (and .svg), or fig_likelihood_confidence_theory.* for
       --norm theory (each readout scaled by its theoretical range instead of its range over the sweep)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from bounded_optimal_confidence.plotting.readout_traces import (W, draw_traces, likelihoods, max_prob, neg_entropy,
                                                                parse_norm, posterior, print_traces, prob_difference,
                                                                style_axis, top2_neg_entropy)
from bounded_optimal_confidence.plotting.style import (AXIS, CATEGORY_LABELS, CORAL, AQUA, DARK, DPI, INK, LABEL_SIZE,
                                                       LINE_WIDTH, PURPLE, STATE_COLORS, TICK_SIZE, TITLE_SIZE,
                                                       use_paper_style)

FIG_DIR = Path(__file__).resolve().parent / "figures"

TASKS = {
    "2AFC": {"mu": (-1.0, 1.0), "sigma": 1.0},
    "3AFC": {"mu": (-2.0, 0.0, 2.0), "sigma": 1.5},
}

OVERLAP_DASH = (0, (3, 3))   # for a readout drawn exactly on top of another

READOUT_COLORS = {
    "Entropy model": PURPLE,
    "Max model": AQUA,
    "Difference model": CORAL,
    "Entropy Top2 model": DARK,
}


def symmetric_ticks(w):
    """Three x ticks (-s, 0, s), with s a round number inside the plotted range."""
    lim = min(abs(w[0]), abs(w[-1]))
    mag = 10 ** np.floor(np.log10(lim))
    s = max(m * mag for m in (1, 2, 4, 5) if m * mag <= lim)
    return [-s, 0, s]


def restyle(ax, w, ylabel):
    """Apply the shared style on top of whatever style_axis / draw_traces set."""
    for side, spine in ax.spines.items():
        spine.set_visible(side in ("left", "bottom"))
        spine.set_color(AXIS)
        spine.set_linewidth(0.8)
    ax.grid(False)
    ax.tick_params(colors=AXIS, labelcolor=INK, labelsize=TICK_SIZE, width=0.8, length=5)
    ax.set_xlim(w[0], w[-1])
    ax.set_xticks(symmetric_ticks(w))
    ax.yaxis.set_major_locator(MaxNLocator(3, steps=[1, 2, 5, 10]))
    ax.set_xlabel("$y$", fontsize=LABEL_SIZE, color=INK)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE, color=INK)


def draw_likelihoods(ax, w, mu, sigma):
    lik = likelihoods(w, mu, sigma)
    for k in range(len(mu)):
        ax.plot(w, lik[:, k], color=STATE_COLORS[k], lw=LINE_WIDTH, solid_capstyle="round")
    style_axis(ax, w)
    restyle(ax, w, "$p(y\\,|\\,x)$")
    ax.set_ylim(0, None)
    return lik


def recolour_readouts(ax):
    """Give each readout its shared colour, solid, at the common line width. A readout that lies exactly on an
    earlier one is dashed, so the one underneath shows through the gaps."""
    drawn = []
    for line in ax.get_lines():
        colour = READOUT_COLORS.get(line.get_label())
        if colour is None:
            continue
        y = line.get_ydata()
        hidden = any(np.allclose(y, other) for other in drawn)
        line.set(color=colour, linestyle=OVERLAP_DASH if hidden else "-", linewidth=LINE_WIDTH, marker="None")
        drawn.append(y)


def confidence_traces(post, task):
    k = post.shape[1]
    traces = [
        ("Entropy model", neg_entropy(post), (-np.log(k), 0)),
        ("Difference model", prob_difference(post), (0, 1)),
        ("Max model", max_prob(post), (1 / k, 1)),   # after Difference: dashed on top where the two coincide
    ]
    if task == "3AFC":
        traces.append(("Entropy Top2 model", top2_neg_entropy(post), (-np.log(2), 0)))
    return traces


def main():
    norm = parse_norm()
    use_paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))

    for row, (task, params) in enumerate(TASKS.items()):
        ax_lik, ax_conf = axes[row]
        lik = draw_likelihoods(ax_lik, W, params["mu"], params["sigma"])
        print(f"{task} likelihood integrals: {np.round(np.trapezoid(lik, W, axis=0), 4)}")

        post = posterior(W, params["mu"], params["sigma"], np.full(len(params["mu"]), 1 / len(params["mu"])))
        scaled = draw_traces(ax_conf, W, confidence_traces(post, task), norm)
        recolour_readouts(ax_conf)
        restyle(ax_conf, W, "confidence")
        print(f"{task} confidence:")
        print_traces(W, scaled)

        # row label, left of the likelihood panel's y-label
        ax_lik.annotate(task, xy=(0, 0.5), xycoords="axes fraction", xytext=(-118, 0), textcoords="offset points",
                        rotation=90, ha="center", va="center", fontsize=TITLE_SIZE)

    # one legend per column, stacked vertically, centred above the top panel; both sit at the same height
    legend_kw = dict(loc="lower center", bbox_to_anchor=(0.5, 1.04), ncol=1, handlelength=1.5, borderaxespad=0)
    for ax, entries in ((axes[0, 0], CATEGORY_LABELS), (axes[0, 1], READOUT_COLORS)):
        handles = [Line2D([], [], color=c, lw=LINE_WIDTH, label=name) for name, c in entries.items()]
        ax.legend(handles=handles, **legend_kw)

    for ax, letter in zip(axes.flat, "ABCD"):
        # centred under the panel, below the x-axis label
        ax.annotate(letter, xy=(0.5, 0), xycoords="axes fraction", xytext=(0, -82), textcoords="offset points",
                    fontsize=TITLE_SIZE, fontweight="bold", ha="center", va="top")

    fig.tight_layout(h_pad=3)
    FIG_DIR.mkdir(exist_ok=True)
    stem = FIG_DIR / ("fig_likelihood_confidence" + ("_theory" if norm == "theory" else ""))
    for suffix in (".png", ".svg"):   # SVG is vector, so dpi only affects any rasterised parts
        out = stem.with_suffix(suffix)
        fig.savefig(out, dpi=DPI, bbox_inches="tight")
        print(f"saved {out}")


if __name__ == "__main__":
    main()
