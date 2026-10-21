"""Figures of the Bayesian model selection over the ten models: the results table and the PXP bar chart.

    python scripts/plot_bms.py --paper      -> figures/paper/bms_table.png, figures/paper/bms_pxp_bars.png
    python scripts/plot_bms.py              -> figures/runs/...   (the newest run of each model)
    python scripts/plot_bms.py --smoke      -> figures/smoke/...

The numbers are those of scripts/bayesian_model_selection.py (`stats.tables.bms_table`).
"""
import argparse

import matplotlib
matplotlib.use("Agg")

from bounded_optimal_confidence.paths import FIGURES_DIR
from bounded_optimal_confidence.plotting.bms_figures import pxp_bar_figure, table_figure
from bounded_optimal_confidence.plotting.style import use_paper_style
from bounded_optimal_confidence.runs import add_source_arguments, fits_from_args, source_tag
from bounded_optimal_confidence.stats.tables import bms_table


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_source_arguments(p)
    a = p.parse_args()
    tab, bor = bms_table(fits_from_args(a))
    out = FIGURES_DIR / source_tag(a)
    use_paper_style()
    table_figure(tab, bor, out / "bms_table.png")
    pxp_bar_figure(tab, bor, out / "bms_pxp_bars.png")


if __name__ == "__main__":
    main()
