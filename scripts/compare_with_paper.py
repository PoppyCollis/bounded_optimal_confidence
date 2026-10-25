"""Compare a new run with the paper's stored result, column by column, and the figures drawn from it, pixel by pixel.

    python scripts/compare_with_paper.py rate-utility results/rate_utility/rateutil_<date>_<time>
    python scripts/compare_with_paper.py shared-attention results/shared_attention/optimise_attention_discrete_Bsweep_<...>
    python scripts/compare_with_paper.py fits [--models diff ent ...]        # the newest run of each model vs results/paper_fits
    python scripts/compare_with_paper.py figures new.png ref.png [new.png ref.png ...]

The three check_paper_* scripts re-solve a few cells with this package and compare them with the stored run; this
script compares a *finished* run (made by rate_utility_sweep.py, optimise_attention_discrete.py --sweep-B or
fit_model.py / fit_all_models.py) with the stored one. The runs are deterministic or Monte Carlo on fixed keyed
streams, so on the pinned numpy / scipy every compared value is expected to be identical. Exits non-zero on any
difference, printing the columns that differ and the largest difference in each.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bounded_optimal_confidence.bounded_rational import paper_run as ru_paper
from bounded_optimal_confidence.models import MODELS, get_model
from bounded_optimal_confidence.runs import newest_run_dir, paper_run_dir, read_fit
from bounded_optimal_confidence.shared_attention import paper_run as sa_paper

FIT_COLUMNS = ("sig_x_index", "sig_x", "log10_alpha", "log10_beta", "b1", "b2", "b3", "lapse", "refit_profile_nll",
               "nll", "edge", "k", "aic")


def _report(result, key, worst, what):
    """Print a per-row table and a verdict for a `compare` result; return True if everything is reproduced."""
    body = result.drop(columns=key)
    per_row = body.all(axis=1)
    bad_cols = list(body.columns[~body.all(axis=0)])
    print(f"{what}: {int(per_row.sum())} of {len(per_row)} rows identical in all {body.shape[1]} compared columns")
    if bad_cols:
        print(f"  columns that differ: {bad_cols}")
        print(f"  largest differences: {worst}")
        with pd.option_context("display.max_rows", 60, "display.width", 200):
            print(result[~per_row][key + bad_cols].head(40).to_string(index=False))
    return bool(per_row.all())


def _log_blocks(path, heads=("[run summary]", "[checks")):
    """The blocks of a config log whose header starts with one of `heads` (the last of each kind)."""
    text = Path(path).read_text()
    blocks = re.split(r"\n(?=\[)", text)
    out = {}
    for b in blocks:
        for h in heads:
            if b.startswith(h):
                out[h] = b.strip()
    return out


def rate_utility(run_dir):
    new = pd.read_csv(Path(run_dir) / "curves.csv", float_precision="round_trip")
    stored = ru_paper.load_paper_curves()
    print(f"new    {run_dir}: {len(new)} rows, {new.shape[1]} columns")
    print(f"stored {ru_paper.PAPER_RUN_DIR}: {len(stored)} rows, {stored.shape[1]} columns")
    ok = list(new.columns) == list(stored.columns)
    if not ok:
        print(f"  COLUMNS DIFFER: new-only {set(new.columns) - set(stored.columns)}, "
              f"stored-only {set(stored.columns) - set(new.columns)}")
    if len(new) != len(stored):
        print("  ROW COUNTS DIFFER")
        ok = False
    result = ru_paper.compare(new, stored)
    ok &= _report(result, ru_paper.KEY, ru_paper.worst_differences(new, stored), "curves.csv")
    for name, path in (("new", Path(run_dir) / "config_log.txt"), ("stored", ru_paper.PAPER_RUN_DIR / "config_log.txt")):
        print(f"\n--- {name} config_log.txt ---")
        for b in _log_blocks(path).values():
            print(b)
    return ok


def shared_attention(run_dir):
    new = pd.read_csv(Path(run_dir) / "sweep.csv", float_precision="round_trip")
    stored = sa_paper.load_paper_sweep()
    print(f"new    {run_dir}: {len(new)} rows, {new.shape[1]} columns")
    print(f"stored {sa_paper.PAPER_RUN_DIR}: {len(stored)} rows, {stored.shape[1]} columns")
    ok = list(new.columns) == list(stored.columns) and len(new) == len(stored)
    if not ok:
        print("  COLUMNS OR ROW COUNTS DIFFER")
    result = sa_paper.compare(new, stored)
    ok &= _report(result, sa_paper.KEY, sa_paper.worst_differences(new, stored),
                  "sweep.csv (seconds, peak_gb not compared)")
    merged = new.merge(stored, on=sa_paper.KEY, suffixes=("", "_stored"))
    print(f"(timing) new {new['seconds'].sum():.0f} s in all, stored {merged['seconds_stored'].sum():.0f} s; "
          f"peak memory new {new['peak_gb'].max():.2f} GB, stored {merged['peak_gb_stored'].max():.2f} GB")
    return ok


def _read_clean(run_dir):
    """clean.csv with the paper files' original column names mapped to this package's."""
    df = pd.read_csv(Path(run_dir) / "clean.csv", float_precision="round_trip")
    df = df.rename(columns={"pid": "sig_x_index"})
    return df.drop(columns=[c for c in ("t", "chirality", "rule", "model_name") if c in df])


def _frame_compare(new, stored, key, columns):
    """One boolean per (row, column): equal, or both NaN."""
    merged = new.merge(stored, on=key, how="outer", suffixes=("", "_stored"), indicator=True)
    out = merged[key].copy()
    out["in_both"] = (merged["_merge"] == "both").to_numpy()
    worst = {}
    for c in columns:
        if c not in new.columns:
            continue
        a, b = merged[c], merged[c + "_stored"]
        if a.dtype.kind in "fiub" and b.dtype.kind in "fiub":
            out[c] = ((a.to_numpy() == b.to_numpy()) | (a.isna() & b.isna()).to_numpy()) & out["in_both"]
            d = np.abs(a.to_numpy(dtype=float) - b.to_numpy(dtype=float))
            d = d[~np.isnan(d)]
            if len(d) and d.max() > 0:
                worst[c] = float(d.max())
        else:
            out[c] = ((a.astype(str) == b.astype(str)) | (a.isna() & b.isna())) & out["in_both"]
    return out, worst


def fits(models, run_dirs=None):
    ok = True
    rows = []
    for name in models:
        model = get_model(name)
        run = Path(run_dirs[name]) if run_dirs and name in run_dirs else newest_run_dir(name)
        if run is None:
            print(f"{name}: no finished run under results/fits")
            ok = False
            continue
        new, stored = read_fit(run, name), read_fit(paper_run_dir(name), name)
        cols = [c for c in FIT_COLUMNS if c in stored.columns]
        res, worst = _frame_compare(new, stored, ["subject"], cols)
        same = res[cols].all(axis=1) & (len(new) == len(stored))
        cell = ["subject", "sig_x_index", "log10_alpha"] + (["log10_beta"] if model.fits_beta else [])
        cn, cs = _read_clean(run), _read_clean(paper_run_dir(name))
        ccols = [c for c in ("sig_x", "refit_profile_nll", "nll", "b1", "b2", "b3", "lapse") if c in cs.columns]
        cres, cworst = _frame_compare(cn, cs, cell, ccols)
        csame = cres[ccols].all(axis=1) & cres["in_both"]
        rows.append({"model": name, "run": run.name, "subjects": len(new), "best identical": int(same.sum()),
                     "max |d nll|": worst.get("nll", 0.0), "max |d aic|": worst.get("aic", 0.0),
                     "clean rows new/stored": f"{len(cn)}/{len(cs)}", "clean identical": int(csame.sum()),
                     "clean max |d nll|": cworst.get("nll", 0.0)})
        this_ok = bool(same.all()) and len(new) == len(stored) and bool(csame.all()) and len(cn) == len(cs)
        if not this_ok:
            print(f"\n{name}: DIFFERS. fit_results columns differing: "
                  f"{[c for c in cols if not res[c].all()]}, worst {worst}; "
                  f"clean columns differing: {[c for c in ccols if not cres[c].all()]}, worst {cworst}")
            bad = res[~res[cols].all(axis=1)]
            if len(bad):
                print(bad.to_string(index=False))
        ok &= this_ok
    print()
    with pd.option_context("display.width", 250):
        print(pd.DataFrame(rows).to_string(index=False))
    return ok


def figures(paths):
    import matplotlib.image as mpimg
    ok = True
    print(f"{'new':60s} {'reference':60s} {'shape':>16s} {'max |d|':>8s} {'pixels differing':>17s}")
    for new, ref in zip(paths[::2], paths[1::2]):
        a, b = mpimg.imread(new), mpimg.imread(ref)
        if a.shape != b.shape:
            print(f"{new:60s} {ref:60s} SHAPES DIFFER {a.shape} vs {b.shape}")
            ok = False
            continue
        d = np.abs(a.astype(float) - b.astype(float))
        n_bad = int((d.max(axis=-1) > 0).sum()) if d.ndim == 3 else int((d > 0).sum())
        print(f"{new:60s} {ref:60s} {str(a.shape):>16s} {d.max():8.4g} {n_bad:17d}")
        ok &= n_bad == 0
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="what", required=True)
    s = sub.add_parser("rate-utility"); s.add_argument("run_dir")
    s = sub.add_parser("shared-attention"); s.add_argument("run_dir")
    s = sub.add_parser("fits")
    s.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    s.add_argument("--run", nargs="+", default=None, metavar="MODEL=DIR", help="a specific run folder for a model")
    s = sub.add_parser("figures"); s.add_argument("paths", nargs="+", help="pairs: new.png ref.png ...")
    a = p.parse_args()
    if a.what == "rate-utility":
        ok = rate_utility(a.run_dir)
    elif a.what == "shared-attention":
        ok = shared_attention(a.run_dir)
    elif a.what == "fits":
        run_dirs = dict(r.split("=", 1) for r in a.run) if a.run else None
        ok = fits(a.models, run_dirs)
    else:
        if len(a.paths) % 2:
            raise SystemExit("figures takes pairs: new.png ref.png ...")
        ok = figures(a.paths)
    print("\nREPRODUCED EXACTLY" if ok else "\nDIFFERENCES FOUND")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
