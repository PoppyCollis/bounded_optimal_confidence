"""Pooled model comparison (exp 1, configs 1-4, one parameter set per subject): the delta-AIC bar chart.

Reads the paper's fit of each of the ten models (`results/paper_fits/<model>/fit_results.csv`, no fitting) and sums
NLL and AIC over the 13 subjects. Deltas are model minus Diff, so positive is worse than Diff. The hard models have
k = 6 (b1 b2 b3 lapse sig_x alpha) and the fitted-beta models k = 7 (+ log10 beta). The bars carry 95% bootstrapped
confidence intervals (`stats.bootstrap.bootstrap_ci`): the 13 per-subject delta AICs resampled with replacement and
summed. The bars stand in registry order (`models.MODELS`), named by their short labels; colour marks the group
(no attention grey, attention green) and opacity the kind of attention (hard, fitted beta).

From `ITBR_confidence/paper_plots/fig_pooled_aic.py`: `summary` and the bars, error bars, limits and ticks of
`bar_figure` are unchanged; the bar layout and the legends are `plotting/model_bars.py`, which was written from the
same function. The fits are read as that script read them (pandas' default float parsing, not the round-trip parsing
of `runs.read_fit`), so the numbers are the ones its figure was drawn from. Its AIC table figure is not carried over.

Run: python paper_plots/fig_pooled_aic.py
    -> paper_plots/figures/pooled_aic_bars.png (and .svg)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from bounded_optimal_confidence.models import MODELS
from bounded_optimal_confidence.plotting.model_bars import ATTENTION_LABEL, add_legends, group_of, model_bars
from bounded_optimal_confidence.plotting.style import INK, use_paper_style
from bounded_optimal_confidence.runs import paper_run_dir
from bounded_optimal_confidence.stats.bootstrap import bootstrap_ci

FIG_DIR = Path(__file__).resolve().parent / "figures"
REFERENCE = "diff"


def paper_fit(model):
    """fit_results.csv of the paper's run of this model, indexed by subject."""
    path = paper_run_dir(model) / "fit_results.csv"
    print(f"  {path}")
    return pd.read_csv(path).set_index("subject")


def summary():
    """One row per model: summed NLL / AIC, deltas against REFERENCE (with the bootstrapped 95% CI of the summed
    delta AIC) and the subjects it beats REFERENCE for."""
    fits = {name: paper_fit(name) for name in MODELS}
    ref = fits[REFERENCE]
    rows = []
    for name, model in MODELS.items():
        f = fits[name]
        d_aic = f.aic - ref.aic
        ci_lo, ci_hi = bootstrap_ci(d_aic, stat=np.sum)
        rows.append({"model": name, "label": model.label, "bar": model.short_label,
                     "subgroup": ATTENTION_LABEL.get(model.attention), "group": group_of(model),
                     "k": int(f.k.iloc[0]), "nll": f.nll.sum(), "aic": f.aic.sum(), "ci_lo": ci_lo, "ci_hi": ci_hi,
                     "d_nll": (f.nll - ref.nll).sum(), "d_aic": d_aic.sum(), "better": int((d_aic < 0).sum()),
                     "n": len(f)})
    return pd.DataFrame(rows)


def bar_figure(tab, path):
    fig, ax, x = model_bars(tab.model, tab.d_aic, f"$\\Delta$AIC relative to {MODELS[REFERENCE].label}")
    # the reference is 0 by definition, so its slot is left empty: no bar and no error bar
    keep = tab.model != REFERENCE
    ax.errorbar(x[keep], tab.d_aic[keep],
                yerr=[(tab.d_aic - tab.ci_lo)[keep], (tab.ci_hi - tab.d_aic)[keep]],
                fmt="none", ecolor=INK, elinewidth=1.5, capsize=6, capthick=1.5, zorder=3)
    ax.set_ylim(min(0, tab.ci_lo.min()), tab.ci_hi.max() * 1.03)    # below 0 only if a whisker goes there
    ax.set_yticks([0, 5000, 10000])
    add_legends(fig, ax)
    for suffix in (".png", ".svg"):   # SVG is vector; dpi only affects any rasterised parts
        out = path.with_suffix(suffix)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"saved {out}")


def main():
    print("runs:")
    tab = summary()
    print(tab.drop(columns=["model", "bar", "subgroup", "group"])
          .to_string(index=False, float_format=lambda v: f"{v:.1f}"))
    use_paper_style()
    FIG_DIR.mkdir(exist_ok=True)
    bar_figure(tab, FIG_DIR / "pooled_aic_bars.png")


if __name__ == "__main__":
    main()
