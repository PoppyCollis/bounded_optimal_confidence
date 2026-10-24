"""The paper figures' shared style: palette, text sizes and matplotlib defaults.

Call `use_paper_style()` once before drawing. It sets the rcParams: sans-serif text in the ink colour, axis lines and
ticks in the axis grey, frameless legends. Nothing is changed on import.
"""
import matplotlib.pyplot as plt

BLUE, ORANGE, GREEN = "#378ADD", "#EF9F27", "#1D9E75"   # the three categories; green is also the accent colour
GREY = "#B4B2A9"                                        # the baseline (no attention) colour
PURPLE, AQUA, CORAL, DARK = "#7F77DD", "#2BC4B6", "#D85A30", "#444441"   # readouts
INK = "#2C2C2A"                                         # text
AXIS = "#5F5E5A"                                        # axis lines and ticks

STATE_COLORS = [BLUE, ORANGE, GREEN]                    # the categories, in order
CATEGORY_LABELS = {f"Category {k + 1}": c for k, c in enumerate(STATE_COLORS)}

LABEL_SIZE = 24
TICK_SIZE = 20
LEGEND_SIZE = 20
TITLE_SIZE = 24
LINE_WIDTH = 2.5
DPI = 300


def use_paper_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": TICK_SIZE,
        "axes.labelsize": LABEL_SIZE,
        "axes.labelcolor": INK,
        "axes.labelpad": 10,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "xtick.color": AXIS,
        "ytick.color": AXIS,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.fontsize": LEGEND_SIZE,
        "legend.frameon": False,
        "text.color": INK,
    })


def save(fig, path):
    """Write the figure at the paper's resolution, cropped to its content, and say where."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")
