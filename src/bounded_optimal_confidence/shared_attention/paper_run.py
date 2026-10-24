"""The stored B sweep the paper's p(x|phi) figure reads, and how a re-solved (B, seed) job is compared with it."""
import csv
import io

import numpy as np
import pandas as pd

from bounded_optimal_confidence.paths import PAPER_SHARED_ATTENTION_DIR

PAPER_RUN_DIR = PAPER_SHARED_ATTENTION_DIR / "optimise_attention_discrete_Bsweep_recentred_n10000_sig64_20261012_150128"
# The sweep's configuration (what `run_sweep` passes to every job for the paper's run).
PAPER_CFG = dict(SIGMA=64.0, SIGMA_RANGE=None, N_WORLDS=10000, N_T=201, WORLD_PRIOR="recentred")
KEY = ["B", "seed"]
TIMING = ("seconds", "peak_gb")     # wall time and peak memory of the job: not reproducible, not compared


def load_paper_sweep(run_dir=PAPER_RUN_DIR):
    """sweep.csv with the floats parsed bit-exactly, so a re-solved value can be compared with ==."""
    return pd.read_csv(run_dir / "sweep.csv", float_precision="round_trip")


def as_written(rows):
    """Rows from `_sweep_job`, taken through the same csv.DictWriter round trip the sweep writes them with."""
    buf = io.StringIO()
    wr = csv.DictWriter(buf, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
    return pd.read_csv(io.StringIO(buf.getvalue()), float_precision="round_trip")


def compare(new, stored):
    """Every non-timing column of `new` against the stored row with the same (B, seed).

    Returns a DataFrame with the key columns and one boolean column per compared column (NaN matches NaN).
    `result.drop(columns=KEY).all(axis=None)` is True when every row is reproduced exactly.
    """
    merged = new.merge(stored, on=KEY, how="left", suffixes=("", "_stored"), validate="one_to_one")
    out = merged[KEY].copy()
    for c in new.columns:
        if c in KEY or c in TIMING:
            continue
        a, b = merged[c], merged[c + "_stored"]
        if a.dtype.kind in "fiub" and b.dtype.kind in "fiub":
            out[c] = (a.to_numpy() == b.to_numpy()) | (a.isna() & b.isna()).to_numpy()
        else:
            out[c] = (a.astype(str) == b.astype(str)) | (a.isna() & b.isna())
    return out


def worst_differences(new, stored):
    """{column: max |new - stored|} over the numeric non-timing columns that differ."""
    merged = new.merge(stored, on=KEY, how="left", suffixes=("", "_stored"), validate="one_to_one")
    out = {}
    for c in new.columns:
        if c in KEY or c in TIMING or new[c].dtype.kind not in "fiu" or merged[c + "_stored"].dtype.kind not in "fiu":
            continue
        d = np.abs(merged[c].to_numpy(dtype=float) - merged[c + "_stored"].to_numpy(dtype=float))
        d = d[~np.isnan(d)]
        if len(d) and d.max() > 0:
            out[c] = float(d.max())
    return out
