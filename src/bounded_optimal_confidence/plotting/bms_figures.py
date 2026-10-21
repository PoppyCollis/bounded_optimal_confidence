"""The Bayesian model selection figures: a table of the results and a bar chart of the protected exceedance
probabilities. Both take the table of `stats.tables.bms_table` and its BOR."""
import matplotlib.pyplot as plt

from bounded_optimal_confidence.plotting.model_bars import add_legends, model_bars
from bounded_optimal_confidence.plotting.style import AXIS, LEGEND_SIZE, save

TABLE_INK, TABLE_MUTED, TABLE_RULE = "black", "#555555", "#cccccc"


def table_figure(tab, bor, path):
    """Model, k, summed AIC, subjects best, frequency, XP and PXP, one row per model, the BOR underneath."""
    cols = ["Model", "k", "AIC", "Subjects\nbest", "Frequency", "XP", "PXP"]
    cells = [[r.label, str(r.k), f"{r.aic:.1f}", f"{r.best}/{r.n}", f"{r.frequency:.3f}", f"{r.xp:.3f}",
              f"{r.pxp:.3f}"] for r in tab.itertuples()]
    xs = [0.0, 0.42, 0.54, 0.65, 0.79, 0.90, 1.0]              # Model: left edge; the rest: right edges
    n = len(cells)
    fig, ax = plt.subplots(figsize=(11.5, 0.5 * n + 1.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(n + 0.5, -1.7)                                  # header at y = -1, one row per y = 0..n-1, note at n
    ax.axis("off")
    for x, c in zip(xs, cols):
        ax.text(x, -1.0, c, ha="left" if x == 0 else "right", va="center", color=TABLE_MUTED, weight="bold",
                fontsize=12, linespacing=1.15)
    for y, row in enumerate(cells):
        for x, v in zip(xs, row):
            ax.text(x, y, v, ha="left" if x == 0 else "right", va="center", color=TABLE_INK, fontsize=12)
    ax.hlines(-0.5, 0, 1, color=TABLE_MUTED, lw=1.0)
    ax.hlines([y + 0.5 for y in range(n)], 0, 1, color=TABLE_RULE, lw=0.6)
    ax.text(0, n, f"Bayesian omnibus risk {bor:.2g}.  {tab.n.iloc[0]} subjects, {n} models, "
            "flat Dirichlet prior ($\\alpha_0 = 1$ per model).", ha="left", va="center", color=TABLE_MUTED,
            fontsize=12)
    save(fig, path)


def pxp_bar_figure(tab, bor, path):
    """PXP per model on the shared bar layout, with the chance level 1 / K dashed and the BOR in the corner."""
    fig, ax, _x = model_bars(tab.model, tab.pxp, "Protected exceedance probability")
    ax.axhline(1 / len(tab), color=AXIS, lw=1.2, ls="--", zorder=0)     # chance: every model equally common
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.text(0.99, 0.98, f"BOR = {bor:.2g}", transform=ax.transAxes, ha="right", va="top", fontsize=LEGEND_SIZE)
    add_legends(fig, ax)
    save(fig, path)
