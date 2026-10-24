"""Decision and confidence of the Difference and Entropy Top-2 models on a bar plot of option probabilities.

4-panel figure: rows are the Difference / Entropy Top-2 models, columns are Decision / Confidence.
    A  Difference model, decision: three options, the chosen (tallest) bar bordered and labelled
    B  Difference model, confidence: the tallest bar's height carried across to the second tallest, an arrow
       spanning the gap between them
    C  Entropy Top-2 model, decision: the two most likely options, renormalised, the chosen one bordered
    D  Entropy Top-2 model, confidence: a dashed box round the two renormalised bars

From `ITBR_confidence/paper_plots/paper_plots_4_new.py`: `plot_bars` and the panel figure unchanged; its other two
figures (`two_option_confidence`, `posterior_decision_criterion`) are not carried over. Its style block is the
shared one of `plotting/style.py` plus the keys it set beyond that (title size, hidden top/right spines, editable
text in PDF/PS output).

Run: python paper_plots/fig_decision_confidence_panels.py
    -> paper_plots/figures/decision_confidence_panels.png (and .svg)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Rectangle

from bounded_optimal_confidence.plotting.style import (AXIS, BLUE, CORAL, DPI, GREEN, INK, ORANGE, TICK_SIZE,
                                                       TITLE_SIZE, use_paper_style)

FIG_DIR = Path(__file__).resolve().parent / "figures"

DASHES = (0, (4, 3))  # dash pattern of the confidence readout marks


def apply_style():
    """The shared paper style, plus what this figure's script set beyond it."""
    use_paper_style()
    plt.rcParams.update({
        "axes.titlesize": TITLE_SIZE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,  # keeps text editable in Illustrator/Acrobat
        "ps.fonttype": 42,
    })


def save(fig, name):
    """PNG and, beside it, SVG (vector, so dpi only matters for any rasterised parts)."""
    for ext in ("png", "svg"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=DPI, bbox_inches="tight")
        print(f"saved {FIG_DIR / f'{name}.{ext}'}")


# Categories are numbered (not lettered) so they can't be confused with the panel letters A-D,
# and coloured as Category 1/2/3 in the likelihood figure.
categories_all = ["1", "2", "3"]
colours_all = [BLUE, ORANGE, GREEN]
edges_all = ["#185FA5",ORANGE, "#0F6E56"]  # darker shade of each colour, for the chosen bar's border
values_all = [0.2, 0.5, 0.3]
width = 0.6
FAINT_ALPHA = 0.3    # fill opacity of every bar


def plot_bars(ax, two_option=False, chosen_idx=None, highlight=None):
    """Bar plot of option probabilities.

    two_option: if True, drop the first option and renormalise so the other two sum to 1.
    chosen_idx: index of the chosen bar (darker border + labelled), or None so all bars look the same.
    highlight: None, "box" (encloses all bars, inside the axes), "dashed_box" (the same box, dashed) or
        "difference" (the tallest bar's height carried across to the second tallest, with an
        arrow spanning the gap between them).
    """
    categories, values = categories_all, np.array(values_all)
    colours, edges = colours_all, edges_all
    if two_option:
        categories, colours, edges = categories[1:], colours[1:], edges[1:]
        values = values[1:] / np.sum(values[1:])

    # every bar has the same faint fill; the chosen one also gets a darker border
    for i, (cat, val, c, e) in enumerate(zip(categories, values, colours, edges)):
        chosen = i == chosen_idx
        ax.bar(cat, val, width=width, color=to_rgba(c, FAINT_ALPHA),
               edgecolor=e if chosen else "none", linewidth=2.5 if chosen else 0, zorder=3)

    ax.set_ylabel("probability")
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.5, 1.0])
    # a little extra side room with two bars, so the "Chosen option" label clears the y-axis
    side = 0.85 if two_option else 0.6
    ax.set_xlim(-side, len(values) - 1 + side)
    ax.tick_params(axis="x", length=0)  # category labels sit under the bars without tick marks

    label_floor = 0.0  # lowest y the "Chosen option" label may sit at (keeps it clear of the box)
    if highlight in ("box", "dashed_box"):
        # Inset from the spines so the box never touches or crosses the axes
        pad = 0.07
        x0, x1 = -width/2 - pad, len(values) - 1 + width/2 + pad
        y0, y1 = 0.015, max(values) + 0.05
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=CORAL, linewidth=2.0,
                               linestyle=DASHES if highlight == "dashed_box" else "-", zorder=5, clip_on=False))
        label_floor = y1 + 0.03
    elif highlight == "difference":
        order = np.argsort(values)[::-1]
        i_top, i_2nd = order[0], order[1]
        h_top, h_2nd = values[i_top], values[i_2nd]
        # the tallest bar's height, carried across to the far edge of the second tallest
        side_sign = np.sign(i_2nd - i_top)
        ax.plot([i_top + side_sign * width/2, i_2nd + side_sign * width/2], [h_top, h_top],
                color=CORAL, linestyle=DASHES, linewidth=1.5, zorder=5)
        # the gap between the two heights is the confidence
        ax.annotate("", xy=(i_2nd, h_top), xytext=(i_2nd, h_2nd), zorder=6,
                    arrowprops=dict(arrowstyle="<->", color=CORAL, lw=2.0, shrinkA=0, shrinkB=0,
                                    mutation_scale=16))

    if chosen_idx is not None:
        top = values[chosen_idx]
        text_y = max(top + 0.05, label_floor)
        ax.annotate("Chosen option", xy=(chosen_idx, top), xytext=(chosen_idx, text_y),
                    ha="center", va="bottom", fontsize=TICK_SIZE,
                    arrowprops=dict(arrowstyle="-", lw=0.8, color=AXIS, shrinkA=0, shrinkB=0))


# 4-panel figure: rows are Difference / Entropy Top-2 models, columns are Decision / Confidence
panels = {
    "A": dict(two_option=False, chosen_idx=1, highlight=None),        # Difference model, decision
    "B": dict(two_option=False, chosen_idx=None, highlight="difference"),  # Difference model, confidence
    "C": dict(two_option=True, chosen_idx=0, highlight=None),         # Entropy Top-2 model, decision
    "D": dict(two_option=True, chosen_idx=None, highlight="dashed_box"),  # Entropy Top-2 model, confidence
}


def main():
    apply_style()
    FIG_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (label, kwargs) in zip(axes.flat, panels.items()):
        plot_bars(ax, **kwargs)
        # centred under the panel, below the category labels
        ax.annotate(label, xy=(0.5, 0), xycoords="axes fraction", xytext=(0, -48), textcoords="offset points",
                    fontsize=TITLE_SIZE, fontweight="bold", ha="center", va="top")
    for ax in axes[:, 1]:
        ax.set_ylabel("")  # axis labels on the first (Decision) column only
    for ax, title in zip(axes[0], ["Decision", "Confidence"]):
        ax.set_title(title, fontsize=TITLE_SIZE, pad=24, color=INK)
    for ax, row_label in zip(axes[:, 0], ["Difference Model", "Entropy Top-2 Model"]):
        # row label, left of the y-axis label
        ax.annotate(row_label, xy=(0, 0.5), xycoords="axes fraction", xytext=(-118, 0), textcoords="offset points",
                    rotation=90, ha="center", va="center", fontsize=TITLE_SIZE)
    plt.tight_layout(h_pad=3)
    save(fig, "decision_confidence_panels")


if __name__ == "__main__":
    main()
