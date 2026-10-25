"""Rate-utility and cost-utility on one config (default 1): 1-step RD, pinned top-2 hierarchy, solved BR hierarchy.

Reads a finished rate-utility sweep (no solving): by default the paper's run,
results/paper_rate_utility/rateutil_20260922_144534. Every curve is swept in its own
inverse temperature (beta for RD, beta3 for the hierarchies, beta1 = beta2 = inf). y = E[U] throughout, labelled
"Accuracy".

One figure, rate_cost_utility_cfg<cfg>_<world>.png (and .svg), two panels:
    A  x = I(W;A), the total rate, the one axis all three share, labelled "Rate I(Y;X)"
    B  x = the charged information cost
                                        RD           I(W;A)/beta
                                        hierarchies  I(W;A|X)/beta3  (serial charges vanish at beta1 = beta2 = inf)
                                    so J = E[U] - x for every curve, and iso-J lines have slope 1.

`--cfg all` writes rate_cost_utility_all_<world>.png (and .svg) instead: 4 x 2 grid, one row per config (number on the
left), Rate column | Cost column, panels lettered A-H underneath, x-axis
labels on the bottom row only.

Run: python paper_plots/fig_rate_cost_utility.py [--cfg 1|2|3|4|all] [--world norm|raw] [--run DIR] [--out-dir DIR]

From ITBR_confidence/paper_plots/fig_rate_cost_utility_cfg1.py. The drawing code is unchanged; the base text sizes it
imported from the other figure scripts are declared below, and the rcParams those scripts set on import are
`plotting.style.use_paper_style()` (the same values).
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

from bounded_optimal_confidence.paths import PAPER_RATE_UTILITY_DIR
from bounded_optimal_confidence.plotting.style import use_paper_style

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
DEFAULT_RUN = str(PAPER_RATE_UTILITY_DIR / "rateutil_20260922_144534")

# The shared paper_plots sizes of the original repo (bayes_1d_readouts: TICK 16, LABEL 19, LEGEND 15;
# fig_likelihood_confidence: TITLE 24). Every text size is that size x TEXT_SCALE, so the ratios between them are kept.
TEXT_SCALE = 1.5
TICK_SIZE = 16 * TEXT_SCALE
LABEL_SIZE = 19 * TEXT_SCALE
LEGEND_SIZE = 15 * TEXT_SCALE
TITLE_SIZE = 24 * TEXT_SCALE

# Okabe-Ito, plus line style so the families never rely on colour alone.
FAMILIES = (
    ("rd", "Rate-Distortion", {"color": "0.6", "lw": 3.5, "ls": "-"}),
    ("top2", "Symmetric Top-2", {"color": "#0072B2", "lw": 3.5, "ls": (0, (5, 2.5))}),
    ("solved", "Bounded-Optimal Attention", {"color": "#E69F00", "lw": 3.5, "ls": "-"}),
)
AXES = {
    "rate": ("I_WA", "Rate $I(Y;X)$"),
    "cost": ("cost_cond", "Information cost"),
}


def load(run_dir, world, cfg):
    df = pd.read_csv(os.path.join(run_dir, "curves.csv"))
    if cfg == "all":
        return df[df["finite"] & (df["world"] == world)]
    df = df[df["finite"] & (df["world"] == world) & (df["stimulus_config"] == cfg)]
    if df.empty:
        raise SystemExit(f"no finite rows for config {cfg}, world {world} in {run_dir}")
    return df


def draw(ax, df, kind):
    xcol, xlab = AXES[kind]
    for fam, label, style in FAMILIES:
        g = df[df["family"] == fam].sort_values("beta3")
        ax.plot(g[xcol], g["E_U"], label=label, zorder=2 if fam == "rd" else 3, **style)
    ax.set_xlabel(xlab, fontsize=LABEL_SIZE)
    ax.set_ylabel("Accuracy", fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE, width=1.2, length=5)
    ax.xaxis.set_major_locator(MaxNLocator(3, steps=[1, 2, 5, 10]))
    ax.yaxis.set_major_locator(MaxNLocator(3, steps=[1, 2, 5, 10]))
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, path):
    """Write the figure as PNG and, beside it, as SVG (vector, so dpi only matters for any rasterised parts)."""
    stem = os.path.splitext(path)[0]
    for ext in (".png", ".svg"):
        fig.savefig(stem + ext, dpi=300, bbox_inches="tight")
        print(f"wrote {stem + ext}")
    plt.close(fig)


def figure(df, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, kind, letter in zip(axes, ("rate", "cost"), "AB"):
        draw(ax, df, kind)
        # centred under the panel, below the x-axis label
        ax.annotate(letter, xy=(0.5, 0), xycoords="axes fraction", xytext=(0, -62 * TEXT_SCALE), textcoords="offset points",
                    fontsize=TITLE_SIZE, fontweight="bold", ha="center", va="top")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
               fontsize=LEGEND_SIZE)
    fig.tight_layout()
    save(fig, path)


def figure_all(df, path):
    cfgs = sorted(df["stimulus_config"].unique())
    fig, axes = plt.subplots(len(cfgs), 2, figsize=(12, 5 * len(cfgs)))
    for i, cfg in enumerate(cfgs):
        d = df[df["stimulus_config"] == cfg]
        for j, kind in enumerate(("rate", "cost")):
            ax = axes[i, j]
            draw(ax, d, kind)
            bottom = i == len(cfgs) - 1
            if not bottom:  # x-axis label on the bottom row only; the letter moves up into its space
                ax.set_xlabel("")
            ax.annotate("ABCDEFGH"[2 * i + j], xy=(0.5, 0), xycoords="axes fraction",
                        xytext=(0, -(62 if bottom else 32) * TEXT_SCALE), textcoords="offset points",
                        fontsize=TITLE_SIZE, fontweight="bold", ha="center", va="top")
        # config number left of the row, outside the y-axis label
        axes[i, 0].annotate(str(cfg), xy=(0, 0.5), xycoords="axes fraction", xytext=(-75 * TEXT_SCALE, 0),
                            textcoords="offset points", fontsize=TITLE_SIZE,
                            ha="right", va="center")
    for ax, name in zip(axes[0], ("Rate", "Cost")):
        ax.set_title(name, fontsize=TITLE_SIZE, pad=20)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
               fontsize=LEGEND_SIZE)
    fig.tight_layout(h_pad=3)
    save(fig, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", choices=("1", "2", "3", "4", "all"), default="1")
    ap.add_argument("--world", choices=("norm", "raw"), default="norm")
    ap.add_argument("--run", default=DEFAULT_RUN)
    ap.add_argument("--out-dir", default=FIG_DIR, help="where the figure goes (default paper_plots/figures)")
    args = ap.parse_args()
    use_paper_style()
    cfg = args.cfg if args.cfg == "all" else int(args.cfg)
    df = load(args.run, args.world, cfg)
    os.makedirs(args.out_dir, exist_ok=True)
    if cfg == "all":
        figure_all(df, os.path.join(args.out_dir, f"rate_cost_utility_all_{args.world}.png"))
    else:
        figure(df, os.path.join(args.out_dir, f"rate_cost_utility_cfg{cfg}_{args.world}.png"))


if __name__ == "__main__":
    main()
