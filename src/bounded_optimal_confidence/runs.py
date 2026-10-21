"""Reading fit results: the paper's fits (results/paper_fits/<model>/) and new runs (results/fits/<model>_<time>/).

Both are returned with the same columns (the paper files' original names are mapped, see results/paper_fits/README.md):
subject, sig_x_index, sig_x, log10_alpha, [log10_beta], b1, b2, b3, lapse, refit_profile_nll, nll, edge, model, k, aic.
"""
import pandas as pd

from bounded_optimal_confidence.models import MODELS, get_model
from bounded_optimal_confidence.paths import FITS_DIR, PAPER_FITS_DIR


def read_fit(run_dir, model):
    """fit_results.csv of a run folder, one row per subject, sorted by subject."""
    model = get_model(model)
    # round_trip: the stored floats come back bit-exact, so a re-score can be compared with ==
    fit = pd.read_csv(run_dir / "fit_results.csv", float_precision="round_trip")
    fit = fit.rename(columns={"pid": "sig_x_index"})
    fit = fit.drop(columns=[c for c in ("t", "chirality", "rule", "model_name") if c in fit])
    fit["model"] = model.name
    return fit.sort_values("subject").reset_index(drop=True)


def paper_run_dir(model):
    return PAPER_FITS_DIR / get_model(model).name


def newest_run_dir(model, fits_dir=FITS_DIR, smoke=False):
    """The newest finished run of `model` under fits_dir (smoke runs only if smoke=True), or None."""
    name = get_model(model).name
    runs = sorted(d for d in fits_dir.glob(f"{name}_*")
                  if d.name.startswith(f"{name}_smoke_") == smoke and (d / "fit_results.csv").exists())
    return runs[-1] if runs else None


def load_fits(paper=False, fits_dir=FITS_DIR, smoke=False, models=MODELS):
    """{model name: (run folder, fit)} for every model that has a run: the paper's fits, or the newest new runs."""
    out = {}
    for name in models:
        run = paper_run_dir(name) if paper else newest_run_dir(name, fits_dir, smoke)
        if run is not None and (run / "fit_results.csv").exists():
            out[name] = (run, read_fit(run, name))
    return out
