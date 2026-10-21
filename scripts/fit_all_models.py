"""Fit all ten models in turn: Diff, Ent, the hard-attention models, then the soft-attention models.

    python scripts/fit_all_models.py                      # the paper's fits: ~10 h in all
    python scripts/fit_all_models.py --models diff ent
    python scripts/fit_all_models.py --subjects 1 --simntrl 400 --smoke

Each model gets its own run folder under results/fits/; scripts/summarise_fits.py tabulates the newest of each.
"""
import argparse
from pathlib import Path

from bounded_optimal_confidence.experiment import SUBJECTS
from bounded_optimal_confidence.fitting.fit import DEFAULT_N_WORKERS, fit_model
from bounded_optimal_confidence.fitting.settings import SEED, SIMNTRL
from bounded_optimal_confidence.models import MODELS
from bounded_optimal_confidence.paths import FITS_DIR


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    p.add_argument("--subjects", type=int, nargs="+", default=list(SUBJECTS))
    p.add_argument("--n-workers", type=int, default=DEFAULT_N_WORKERS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--simntrl", type=int, default=SIMNTRL)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--out-dir", type=Path, default=FITS_DIR)
    a = p.parse_args()
    for name in a.models:
        print(f"\n=== {name} ===", flush=True)
        fit_model(name, a.subjects, a.n_workers, a.seed, a.simntrl, a.smoke, a.out_dir)


if __name__ == "__main__":
    main()
