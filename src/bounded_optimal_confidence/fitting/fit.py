"""Fit one model to every subject: search -> shortlist -> clean re-score -> best fit per subject.

    1. search      score the grid with the b-triple and lapse profiled out (`search`), on 2,500 replicates per
                   trial and common random numbers across cells
    2. shortlist   each subject's N_REFINE best search cells
    3. clean       per shortlisted cell, re-profile the b-triple and lapse on a fresh draw of 10,000 replicates, then
                   score them on another, independent draw of 100,000 (`tasks.clean_score`)
    4. best        per subject, the shortlisted cell with the lowest clean NLL; AIC = 2 (k + NLL)

A run writes to its own folder: config_log.txt, profile.csv (all search cells), clean.csv (the shortlist, scored),
fit_results.csv (one row per subject) and report.md.
"""
from datetime import datetime

import pandas as pd

from bounded_optimal_confidence.experiment import SUBJECTS
from bounded_optimal_confidence.fitting import report
from bounded_optimal_confidence.fitting.search import search
from bounded_optimal_confidence.fitting.settings import (CLEAN_SIMNTRL_FACTOR, N_REFINE, RANK_SIMNTRL_FACTOR, SEED,
                                                         SIG_X_STRIDE, SIMNTRL, SMOKE_N_REFINE, SMOKE_SIG_X_STRIDE)
from bounded_optimal_confidence.fitting.tasks import CleanTask, clean_score
from bounded_optimal_confidence.models import get_model
from bounded_optimal_confidence.paths import FITS_DIR
from bounded_optimal_confidence.utils.parallel import run_parallel
from bounded_optimal_confidence.utils.provenance import git_commit

DEFAULT_N_WORKERS = 14


def cell_columns(model):
    """The columns that identify a grid cell of this model, per subject."""
    return ["subject", "sig_x_index", "log10_alpha"] + (["log10_beta"] if model.fits_beta else [])


def shortlist(profile, model, n_refine):
    """Each subject's n_refine best search cells (lowest profiled NLL)."""
    short = profile.sort_values("profile_nll").groupby("subject").head(n_refine)
    return short.drop_duplicates(cell_columns(model))


def clean(short, model, n_reps, clean_n_reps, seed, n_workers):
    tasks = [CleanTask(model.name, int(r.subject), int(r.sig_x_index), float(r.log10_alpha),
                       float(r.log10_beta) if model.fits_beta else None, n_reps, clean_n_reps, seed)
             for r in short.itertuples()]
    return pd.DataFrame(run_parallel(clean_score, tasks, n_workers, "clean", every=50))


def best_per_subject(scored, profile, model):
    """Per subject, the cleanly scored cell with the lowest NLL, with its edge flags, k and AIC."""
    best = scored.sort_values("nll").groupby("subject", as_index=False).head(1).sort_values("subject")
    best = best.reset_index(drop=True)
    best["edge"] = report.edge_flags(best, profile, model)
    best["model"], best["k"] = model.name, model.k
    best["aic"] = 2 * (best.k + best.nll)
    return best


def fit_model(model, subjects=SUBJECTS, n_workers=DEFAULT_N_WORKERS, seed=SEED, simntrl=SIMNTRL, smoke=False,
              out_dir=FITS_DIR):
    """Fit `model` (a name or Model) to `subjects`; returns the run folder.

    `smoke` makes a quick end-to-end check rather than a fit: stage A of the coarse-to-fine search takes every
    SMOKE_SIG_X_STRIDE-th sig_x and only SMOKE_N_REFINE cells per subject are clean-scored. Pair it with a small
    `simntrl` (e.g. 400).
    """
    model = get_model(model)
    rank_n, clean_n = max(1, int(RANK_SIMNTRL_FACTOR * simntrl)), CLEAN_SIMNTRL_FACTOR * simntrl
    stride, n_refine = (SMOKE_SIG_X_STRIDE, SMOKE_N_REFINE) if smoke else (SIG_X_STRIDE, N_REFINE)
    started = datetime.now()
    out = out_dir / f"{model.name}_{'smoke_' if smoke else ''}{started:%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    report.write_config_log(out / "config_log.txt", model, subjects, git_commit(), started, seed,
                            (rank_n, simntrl, clean_n), stride, n_refine)

    profile = search(model, subjects, rank_n, seed, n_workers, stride)
    profile.to_csv(out / "profile.csv", index=False)

    scored = clean(shortlist(profile, model, n_refine), model, simntrl, clean_n, seed, n_workers)
    scored.to_csv(out / "clean.csv", index=False)

    best = best_per_subject(scored, profile, model)
    best.to_csv(out / "fit_results.csv", index=False)
    print(report.write_report(out / "report.md", best, model, out.name))
    report.finish_config_log(out / "config_log.txt")
    return out
