"""p(x|phi) and the router cut of the sparse optimum against B / sigma: one panel per structure (one line per
state), and a third panel with the stimulus at which the hard router switches from phi_1 to phi_2 (the
stimulus is w in the code and the sweep files, labelled y in the figure).

Reads one finished B sweep of `generalisation.optimise_attention` (no solving): the newest results folder
`optimise_attention_Bsweep_<prior>_n*_sig<sigma>_*`, whose sweep.csv has one row per (B, seed) with the two
optimal structure priors. Lines are the mean over seeds, the faint dots the seeds. The dashed line at 0.5 is the
weight of the code [[.5, .5, 0], [0, .5, .5]] the optimum tends to.

The cut is known to one step of the sweep's w grid (2.85 = 0.045 sigma): sweep.csv stores the first grid point
routed to the other structure, and the cut plotted is the middle of the grid cell before it, in units of sigma.

Default is the paper's sweep, results/paper_shared_attention/optimise_attention_discrete_Bsweep_recentred_n10000_sig64_20261012_150128
(discrete w; recentred world prior, sigma 64, 10000 worlds, seeds 0, 2, 4, 6, 8, B = 64 .. 2000). Limits of that
sweep, visible in the figure:
  - six B values, so the segment between B / sigma 2 and 3.9 is a straight line across the change of supports;
  - the largest B / sigma is 31, where the middle weight is 0.486: close to 0.5, not on it.

phi_1 is the structure further left (set by `optimise_attention.solve`), so nothing is relabelled here.

Styled to match the other paper figures (text sizes, palette and axes from fig_likelihood_confidence): the states
use the shared state colours (left blue, middle orange, right green), as in the likelihood figures.

Run: python paper_plots/fig_pxgphi_vs_B.py [--run <sweep dir> | --latest [--world-prior recentred]] [--out-dir DIR]
    -> paper_plots/figures/pxgphi_vs_B_<prior>_discrete.png (and .svg); the `_discrete` tag comes from the sweep
       folder's name (a sweep of the continuous script, read with --run, would drop it)

From ITBR_confidence/paper_plots/fig_pxgphi_vs_B.py. The drawing code is unchanged; the style constants it imported
from the other figure scripts come from `plotting.style`, which sets the same rcParams in `use_paper_style()`.
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bounded_optimal_confidence.paths import SHARED_ATTENTION_DIR
from bounded_optimal_confidence.plotting.style import (AXIS, BLUE, DARK, GREEN, INK, LABEL_SIZE, LEGEND_SIZE,
                                                       LINE_WIDTH, ORANGE, TICK_SIZE, TITLE_SIZE, use_paper_style)
from bounded_optimal_confidence.shared_attention.paper_run import PAPER_RUN_DIR

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
RESULTS = str(SHARED_ATTENTION_DIR)      # new sweeps; the paper's own is PAPER_RUN_DIR
STATE_COLORS = [BLUE, ORANGE, GREEN]     # left, middle, right: the state colours of the likelihood figures
use_paper_style()

STATES = ("left", "middle", "right")
STATE_MARKERS = ("o", "s", "^")    # a second cue besides colour
TARGET = 0.5        # the two non-zero weights of [[.5, .5, 0], [0, .5, .5]]
TARGET_COLOR = AXIS
CUT_COLOR = DARK
CUT_LIM = 1         # y range of the cut panel, in sigma
DPI = 300
# One factor on every text size, so they all grow in proportion (1.0 = the same sizes as the other figures)
TEXT_SCALE = 1.3
LABEL, TICK, LEGEND, TITLE = (size * TEXT_SCALE for size in (LABEL_SIZE, TICK_SIZE, LEGEND_SIZE, TITLE_SIZE))
LEGEND = LABEL      # the legend at the axis-label size, so it reads as easily as the labels
TITLE_PAD = 28      # gap between each panel title and its axes, in points

# Re-applied here so this figure keeps the shared look even if an imported module changed the defaults.
plt.rcParams.update({
    "font.family": "sans-serif",
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.labelpad": 10,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "legend.frameon": False,
})


def newest_run(prior):
    dirs = sorted(glob.glob(os.path.join(RESULTS, f"optimise_attention*_Bsweep_{prior}_n*")),
                  key=lambda d: d.rsplit("_", 2)[-2:])   # by the time stamp at the end of the name
    if not dirs:
        raise SystemExit(f"no optimise_attention*_Bsweep_{prior}_* run in {RESULTS}")
    return dirs[-1]


def load(run_dir):
    """(prior, sigma, B / sigma per row, pi[row, structure, state], cut / sigma per row, sorted unique
    B / sigma)."""
    m = re.search(r"Bsweep_(.+)_n\d+_sig([0-9.]+)(-[0-9.]+)?_\d{8}_\d{6}$", os.path.basename(run_dir.rstrip("/")))
    if not m:
        raise SystemExit(f"{run_dir} is not an optimise_attention B sweep folder")
    if m.group(3):
        raise SystemExit(f"{run_dir} drew sigma from a range: B / sigma is not defined for it")
    prior, sigma = m.group(1), float(m.group(2))
    df = pd.read_csv(os.path.join(run_dir, "sweep.csv"))
    pi = np.stack([df[[f"pi{i}_{s}" for s in STATES]].to_numpy() for i in (1, 2)], axis=1)
    assert np.allclose(pi.sum(2), 1), "a p(x|phi) column does not sum to 1"
    x = df["B"].to_numpy() / sigma
    switches = df["switches"].astype(str)
    if switches.str.contains(";").any() or (switches == "nan").any():
        raise SystemExit(f"{run_dir}: a row's router does not switch exactly once")
    step = 2 * (df["B"] + 5 * sigma) / (df["n_grid"] - 1)   # optimise_attention.make_grid
    cut = (switches.astype(float) - step / 2).to_numpy() / sigma
    return prior, sigma, x, pi, cut, np.unique(x)


def log2_ticks(xs):
    """Powers of 2 spanning the sweep; every other one when there would be more than four."""
    ticks = 2.0 ** np.arange(np.floor(np.log2(xs[0])), np.ceil(np.log2(xs[-1])) + 1)
    return ticks[::2] if len(ticks) > 4 else ticks


def figure(x, pi, cut, xs, path):
    mean = np.stack([pi[x == v].mean(0) for v in xs])   # (B, structure, state)
    cut_mean = np.array([cut[x == v].mean() for v in xs])
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.5))
    axes[1].sharey(axes[0])
    axes[1].tick_params(labelleft=False)
    for k, ax in enumerate(axes[:2]):
        ax.axhline(TARGET, color=TARGET_COLOR, lw=1.5, ls=(0, (4, 3)), zorder=1)
        for s, (name, color, marker) in enumerate(zip(STATES, STATE_COLORS, STATE_MARKERS)):
            ax.scatter(x, pi[:, k, s], s=18, color=color, alpha=0.35, lw=0, zorder=2)
            ax.plot(xs, mean[:, k, s], color=color, marker=marker, ms=8, lw=LINE_WIDTH, label=name, zorder=3)
        ax.set_ylim(-0.04, 1.04)
        ax.set_yticks([0, 0.5, 1])
        ax.set_title(rf"$p(x\,|\,\phi_{k + 1})$", fontsize=TITLE, color=INK, pad=TITLE_PAD)
    ax = axes[2]
    ax.axhline(0, color=TARGET_COLOR, lw=1.5, ls=(0, (4, 3)), zorder=1)
    ax.scatter(x, cut, s=18, color=CUT_COLOR, alpha=0.35, lw=0, zorder=2)
    ax.plot(xs, cut_mean, color=CUT_COLOR, marker="D", ms=8, lw=LINE_WIDTH, zorder=3)
    ax.set_ylim(-CUT_LIM, CUT_LIM)
    ax.set_yticks([-CUT_LIM, 0, CUT_LIM])
    ax.set_ylabel(r"$y\,/\,\sigma$", fontsize=LABEL)
    ax.set_title(r"switch-point of $p(\phi\,|\,y)$", fontsize=TITLE, color=INK, pad=TITLE_PAD)
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(log2_ticks(xs))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel(r"$B\,/\,\sigma$", fontsize=LABEL)
        ax.tick_params(labelsize=TICK, width=0.8, length=6, color=AXIS, labelcolor=INK, pad=6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(AXIS)
        ax.spines[["left", "bottom"]].set_linewidth(0.8)
    axes[0].set_ylabel("probability", fontsize=LABEL)
    for ax, letter in zip(axes, "ABC"):
        # panel letter, centred under the panel, below the x-axis label
        ax.annotate(letter, xy=(0.5, 0), xycoords="axes fraction", xytext=(0, -100), textcoords="offset points",
                    fontsize=TITLE, fontweight="bold", ha="center", va="top", color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(w_pad=3)
    mid = 0.5 * (axes[0].get_position().x0 + axes[1].get_position().x1)   # legend over the two p(x|phi) panels
    fig.legend(handles, labels, frameon=False, loc="lower center", bbox_to_anchor=(mid, 1.0), ncol=3,
               fontsize=LEGEND, handlelength=2.0, columnspacing=2.0, markerscale=1.2)
    stem = os.path.splitext(path)[0]
    for ext in (".png", ".svg"):   # SVG is vector; dpi only affects any rasterised parts
        fig.savefig(stem + ext, dpi=DPI, bbox_inches="tight")
        print(f"wrote {stem + ext}")
    plt.close(fig)
    return mean, cut_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(PAPER_RUN_DIR), help="a B sweep folder (default: the paper's sweep)")
    ap.add_argument("--latest", action="store_true", help="instead, the newest sweep under results/shared_attention")
    ap.add_argument("--world-prior", default="recentred", help="with --latest: world prior of the sweep to read")
    ap.add_argument("--out-dir", default=FIG_DIR, help="where the figure goes (default paper_plots/figures)")
    args = ap.parse_args()
    run_dir = newest_run(args.world_prior) if args.latest else args.run
    prior, sigma, x, pi, cut, xs = load(run_dir)
    print(f"{run_dir}\n{prior}, sigma {sigma:g}, {len(xs)} B values, {len(x)} rows")
    os.makedirs(args.out_dir, exist_ok=True)
    tag = "_discrete" if "_discrete_" in os.path.basename(run_dir.rstrip("/")) else ""  # discrete-w sweep
    mean, cut_mean = figure(x, pi, cut, xs, os.path.join(args.out_dir, f"pxgphi_vs_B_{prior}{tag}.png"))
    np.set_printoptions(precision=3, suppress=True)
    print(f"{'B/sigma':>8s}  {'p(x|phi_1)':<22s}{'p(x|phi_2)':<22s}{'cut/sigma':>10s}{'min':>8s}{'max':>8s}"
          f"{'seeds':>6s}")
    for v, m, c in zip(xs, mean, cut_mean):
        print(f"{v:8.2f}  {str(m[0]):<22s}{str(m[1]):<22s}{c:10.4f}{cut[x == v].min():8.4f}"
              f"{cut[x == v].max():8.4f}{(x == v).sum():6d}")


if __name__ == "__main__":
    main()