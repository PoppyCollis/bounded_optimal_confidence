"""Fit one model to the exp 1 data (configs 1-4 pooled, one parameter set per subject).

    python scripts/fit_model.py --model sym_spec_soft
    python scripts/fit_model.py --model diff --subjects 1 2 --n-workers 8
    python scripts/fit_model.py --model diff --subjects 1 --simntrl 400 --smoke      # quick end-to-end check

Writes results/fits/<model>_<date>_<time>/ (see bounded_optimal_confidence.fitting.fit). The paper's runs
took 26-44 min per k = 6 model and about 1.5 h per soft-attention model.
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
    p.add_argument("--model", choices=list(MODELS), required=True)
    p.add_argument("--subjects", type=int, nargs="+", default=list(SUBJECTS))
    p.add_argument("--n-workers", type=int, default=DEFAULT_N_WORKERS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--simntrl", type=int, default=SIMNTRL,
                   help=f"replicates per trial for the refit (default {SIMNTRL}); search 1/4 of it, clean 10x")
    p.add_argument("--smoke", action="store_true", help="quick check: coarser stage A, 3 cells refined per subject")
    p.add_argument("--out-dir", type=Path, default=FITS_DIR)
    a = p.parse_args()
    fit_model(a.model, a.subjects, a.n_workers, a.seed, a.simntrl, a.smoke, a.out_dir)


if __name__ == "__main__":
    main()
