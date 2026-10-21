"""Tabulate the fits of all models: summed NLL and AIC, delta AIC to Diff, and nested-model search checks.

    python scripts/summarise_fits.py --paper        # the paper's fits (results/paper_fits)
    python scripts/summarise_fits.py                # the newest run of each model in results/fits
    python scripts/summarise_fits.py --smoke        # the newest smoke runs

Delta AIC is the model minus Diff, summed over subjects (positive = worse than Diff).

Search checks. A soft-attention model contains its hard twin (beta = inf), and on the three partition repertoires
also Ent (beta = 1: the Bayes posterior with the entropy readout). Its best fit can then never be worse than theirs,
up to Monte Carlo noise in the clean NLL; a subject more than TOL nats worse points to a search failure.
"""
import argparse

from bounded_optimal_confidence.models import MODELS
from bounded_optimal_confidence.runs import add_source_arguments, fits_from_args

REFERENCE = "diff"
TOL = 1.0


def nested_in(name):
    """The fitted models nested in `name` (empty for the models without beta)."""
    model = MODELS[name]
    if not model.fits_beta:
        return []
    return [name.replace("_soft", "_hard")] + (["ent"] if model.repertoire != "sym_top2" else [])


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_source_arguments(p)
    fits = fits_from_args(p.parse_args())

    ref = fits[REFERENCE][1].set_index("subject") if REFERENCE in fits else None
    print(f"\n{'model':18s} {'label':18s} {'k':>2s} {'n':>3s} {'NLL':>9s} {'AIC':>9s} {'dAIC vs Diff':>13s} "
          f"{'better than Diff':>17s}")
    for name, (_run, fit) in fits.items():
        f = fit.set_index("subject")
        line = (f"{name:18s} {MODELS[name].label:18s} {int(f.k.iloc[0]):2d} {len(f):3d} {f.nll.sum():9.1f} "
                f"{f.aic.sum():9.1f}")
        if ref is not None:
            d = (f.aic - ref.aic).dropna()
            line += f" {d.sum():13.1f} {int((d < 0).sum()):>11d}/{len(d)}"
        print(line)

    print(f"\nsearch checks (a subject more than {TOL:g} nat worse than a nested model):")
    for name, (_run, fit) in fits.items():
        for inner in nested_in(name):
            if inner not in fits:
                print(f"  {name} vs {inner}: no {inner} run")
                continue
            d = (fit.set_index("subject").nll - fits[inner][1].set_index("subject").nll).dropna()
            worse = [f"S{s} {v:+.2f}" for s, v in d.items() if v > TOL]
            print(f"  {name:18s} vs {inner:16s} summed dNLL {d.sum():+9.1f}; worse: {', '.join(worse) or 'none'}")


if __name__ == "__main__":
    main()
