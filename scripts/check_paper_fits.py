"""Check that this package reproduces the paper's fits: re-score every subject's best cell and compare.

    python scripts/check_paper_fits.py                          # all models, all subjects (130 cells, a few minutes)
    python scripts/check_paper_fits.py --models diff --subjects 1

For each model and subject, the stored best cell (sig_x, alpha[, beta]) of results/paper_fits/<model> goes through the
clean stage again (`fitting.tasks.clean_score`: refit of the b-triple and lapse on 10,000 replicates per trial, NLL on
100,000) with the run's seed. The fits are Monte Carlo with fixed random streams, so identical code gives identical
numbers: the script reports the largest difference per model and exits non-zero if any value differs.
"""
import argparse
import sys

import numpy as np
import pandas as pd

from bounded_optimal_confidence.experiment import SUBJECTS
from bounded_optimal_confidence.fitting.fit import DEFAULT_N_WORKERS
from bounded_optimal_confidence.fitting.settings import CLEAN_SIMNTRL_FACTOR, SEED, SIMNTRL
from bounded_optimal_confidence.fitting.tasks import CleanTask, clean_score
from bounded_optimal_confidence.models import MODELS, get_model
from bounded_optimal_confidence.runs import load_fits
from bounded_optimal_confidence.utils.parallel import run_parallel

COMPARED = ("nll", "refit_profile_nll", "b1", "b2", "b3", "lapse")


def rescore(models, subjects, n_workers):
    """One row per (model, subject): the stored values and the re-scored ones (suffix _new)."""
    stored, tasks = [], []
    for name, (_run, fit) in load_fits(paper=True, models=models).items():
        model = get_model(name)
        for r in fit[fit.subject.isin(subjects)].itertuples():
            stored.append(r._asdict())
            tasks.append(CleanTask(name, int(r.subject), int(r.sig_x_index), float(r.log10_alpha),
                                   float(r.log10_beta) if model.fits_beta else None, SIMNTRL,
                                   CLEAN_SIMNTRL_FACTOR * SIMNTRL, SEED))
    new = run_parallel(clean_score, tasks, n_workers, "re-score", every=10)
    rows = []
    for old, row in zip(stored, new):
        rows.append({"model": old["model"], "subject": old["subject"],
                     **{c: old[c] for c in COMPARED}, **{f"{c}_new": row[c] for c in COMPARED}})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    p.add_argument("--subjects", type=int, nargs="+", default=list(SUBJECTS))
    p.add_argument("--n-workers", type=int, default=DEFAULT_N_WORKERS)
    a = p.parse_args()
    tab = rescore(a.models, a.subjects, a.n_workers)
    ok = True
    print(f"\n{'model':18s} {'subjects':>8s} {'identical':>9s} {'max |d nll|':>12s} {'max |d b, lapse|':>17s}")
    for name, g in tab.groupby("model", sort=False):
        same = np.all([g[c].to_numpy() == g[f"{c}_new"].to_numpy() for c in COMPARED], axis=0)
        d_nll = (g.nll_new - g.nll).abs().max()
        d_par = max((g[f"{c}_new"] - g[c]).abs().max() for c in ("b1", "b2", "b3", "lapse"))
        ok &= bool(same.all())
        print(f"{name:18s} {len(g):8d} {int(same.sum()):9d} {d_nll:12.3g} {d_par:17.3g}")
    print("\nall re-scored values identical to the paper's fits" if ok else
          "\nSOME VALUES DIFFER from the paper's fits (see the table)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
