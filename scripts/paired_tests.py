"""Paired tests of every model against Diff on the per-subject AIC.

    python scripts/paired_tests.py --paper                  # the paper's fits
    python scripts/paired_tests.py                          # the newest run of each model
    python scripts/paired_tests.py --smoke --reference ent

The difference is model minus reference, subject by subject, so positive means the reference fits better. For
each model: the summed difference (the delta AIC) with its bootstrap 95% CI, the mean and sd, the paired two-tailed
t-test with the 95% CI of the mean and d_z = mean / sd, the exact Wilcoxon signed-rank p, and the number of
subjects the model fits better. docs/statistics.md has the details.
"""
import argparse

from bounded_optimal_confidence.models import MODELS
from bounded_optimal_confidence.runs import add_source_arguments, fits_from_args
from bounded_optimal_confidence.stats.tables import REFERENCE, paired_table


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_source_arguments(p)
    p.add_argument("--reference", choices=list(MODELS), default=REFERENCE)
    a = p.parse_args()
    fits = fits_from_args(a)
    if a.reference not in fits:
        raise SystemExit(f"no fit of the reference model {a.reference}")

    tab = paired_table(fits, a.reference)
    n = tab.n.iloc[0]
    print(f"\nmodel minus {MODELS[a.reference].label}, per-subject AIC, {n} subjects")
    print(f"{'model':18s} {'sum dAIC':>9s} {'95% CI (bootstrap)':>20s} {'mean':>7s} {'sd':>6s} {'t(df)':>8s} "
          f"{'p':>9s} {'95% CI of mean':>17s} {'d_z':>6s} {'Wilcoxon p':>10s} {'better':>7s}")
    for r in tab.itertuples():
        print(f"{r.model:18s} {r.sum:+9.1f} [{r.sum_ci_lo:+8.1f}, {r.sum_ci_hi:+8.1f}] {r.mean:+7.2f} {r.sd:6.2f} "
              f"{r.t:+5.2f}({r.df}) {r.p:9.3g} [{r.ci_lo:+6.2f}, {r.ci_hi:+6.2f}] {r.d_z:+6.2f} {r.wilcoxon_p:10.3g} "
              f"{r.n_better:>4d}/{n}")


if __name__ == "__main__":
    main()
