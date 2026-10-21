"""The pooled fit of one model to every subject (exp 1, configs 1-4, one parameter set per subject).

    settings    every grid and constant
    seeding     the common-random-number streams
    tasks       the work done in worker processes (score grid cells; clean-score a cell)
    search      the grid search: full grid (k 6) or coarse-to-fine (soft attention, k 7)
    fit         search -> shortlist -> clean re-score -> best per subject; writes the run folder
    report      edge flags, config_log.txt, report.md
"""
from bounded_optimal_confidence.fitting.fit import fit_model

__all__ = ["fit_model"]
