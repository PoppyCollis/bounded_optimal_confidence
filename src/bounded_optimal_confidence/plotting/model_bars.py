"""One bar per model, laid out as in the paper's model-comparison figures.

The bars stand in registry order. Colour marks the group (no attention in grey, attention in green), opacity the
kind of attention (hard 0.75, soft 0.3), and a gap separates the groups. The bar names are the short labels
(Diff, Ent, Sym Top-2, ...); two legends above the axes explain colour and opacity.

    fig, ax, x = model_bars(models, heights, ylabel)     # bars, axis styling, bar names
    ... add error bars, reference lines, text ...
    add_legends(fig, ax)                                 # last: the legends are placed from the drawn figure
"""
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
import numpy as np

from bounded_optimal_confidence.models import get_model
from bounded_optimal_confidence.plotting.style import AXIS, GREEN, GREY, INK, LABEL_SIZE, LEGEND_SIZE, TICK_SIZE

BAR_WIDTH = 0.6
GROUP_GAP = 0.6                                                 # extra space, in bar slots, where the group changes
GROUP_COLOR = {"No attention": GREY, "Attention": GREEN}
ATTENTION_ALPHA = {"hard": 0.75, "soft": 0.3}
ATTENTION_LABEL = {"hard": "Hard attention ($\\beta = \\infty$)", "soft": "Soft attention $\\beta$"}
FIGSIZE = (14, 7)


def group_of(model):
    return "No attention" if get_model(model).attention == "none" else "Attention"


def bar_positions(models):
    """x of each bar: consecutive slots, plus GROUP_GAP wherever the (group, attention) pair changes."""
    keys = [(group_of(m), get_model(m).attention) for m in models]
    changes = np.cumsum([i > 0 and keys[i] != keys[i - 1] for i in range(len(keys))])
    return np.arange(len(models)) + GROUP_GAP * changes


def bar_colors(models):
    return [to_rgba(GROUP_COLOR[group_of(m)], ATTENTION_ALPHA.get(get_model(m).attention, 1.0)) for m in models]


def model_bars(models, heights, ylabel, figsize=FIGSIZE):
    """Draw the bars and style the axes. Returns (fig, ax, x), x being the bar positions."""
    x = bar_positions(models)
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x, heights, width=BAR_WIDTH, color=bar_colors(models))
    ax.set_xticks(x)
    ax.set_xticklabels([get_model(m).short_label for m in models], rotation=40, ha="right", rotation_mode="anchor")
    ax.set_xlim(x.min() - BAR_WIDTH, x.max() + BAR_WIDTH)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE, width=0.8, length=5, color=AXIS, labelcolor=INK)
    ax.tick_params(axis="x", length=0)                          # the names sit under the bars without tick marks
    ax.spines[["top", "right"]].set_visible(False)
    return fig, ax, x


def add_legends(fig, ax):
    """Two legends above the axes, side by side: colour = decision model group, opacity = kind of attention.
    The attention swatch stands for both kinds, so it takes the shade halfway between them."""
    mid = np.mean(list(ATTENTION_ALPHA.values()))
    first = ax.legend(handles=[Patch(facecolor=to_rgba(c, mid if g == "Attention" else 1.0), label=g)
                               for g, c in GROUP_COLOR.items()], title="Decision models",
                      loc="lower left", bbox_to_anchor=(0.0, 1.04), borderaxespad=0, frameon=False,
                      fontsize=LEGEND_SIZE, title_fontsize=LEGEND_SIZE, alignment="left")
    ax.add_artist(first)
    first.set_clip_on(False)                                    # else the tight crop treats it as clipped to the axes
    fig.tight_layout()
    fig.canvas.draw()                                           # the second legend sits just right of the first
    right = first.get_window_extent().transformed(ax.transAxes.inverted()).x1
    ax.legend(handles=[Patch(facecolor=to_rgba(GROUP_COLOR["Attention"], a), label=ATTENTION_LABEL[kind])
                       for kind, a in ATTENTION_ALPHA.items()],
              loc="lower left", bbox_to_anchor=(right + 0.05, 1.04), borderaxespad=0, frameon=False,
              fontsize=LEGEND_SIZE)
