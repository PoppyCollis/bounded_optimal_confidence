"""Random-effects Bayesian model selection over the fits: the ten models, then Diff against the Sym Top-2 family.

    python scripts/bayesian_model_selection.py --paper      # the paper's fits
    python scripts/bayesian_model_selection.py              # the newest run of each model
    python scripts/bayesian_model_selection.py --smoke      # the newest smoke runs

Each subject's log model evidence is -AIC / 2. Every subject's best model is treated as a draw from unknown
population frequencies (Stephan et al. 2009; Rigoux et al. 2014; `stats.bms`), which gives per model

    frequency   the expected share of the population best described by the model
    XP          the probability that the model is the most common one
    PXP         XP protected against the null that all models are equally common: the number to report

and the Bayesian omnibus risk (BOR), the probability of that null. The results depend on the model set, because
similar models share subjects. The second part therefore reruns it on Diff and the two Sym Top-2 models alone,
at the model level and at the family level (No attention vs Sym Top-2). docs/statistics.md has the details.
"""
import argparse

from bounded_optimal_confidence.runs import add_source_arguments, fits_from_args
from bounded_optimal_confidence.stats.tables import bms_table, family_bms_tables

FMT = {"aic": "{:.1f}".format, "frequency": "{:.4f}".format, "xp": "{:.4f}".format, "pxp": "{:.4f}".format}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_source_arguments(p)
    fits = fits_from_args(p.parse_args())

    tab, bor = bms_table(fits)
    print(f"\nall {len(tab)} models ({tab.n.iloc[0]} subjects, flat Dirichlet prior, 1 per model)")
    print(tab.drop(columns=["n"]).to_string(index=False, formatters=FMT))
    print(f"Bayesian omnibus risk: {bor:.3g}")

    for title, t, bor in family_bms_tables(fits):
        print(f"\n{title}")
        print(t.to_string(index=False, formatters=FMT))
        print(f"Bayesian omnibus risk: {bor:.3g}")


if __name__ == "__main__":
    main()
