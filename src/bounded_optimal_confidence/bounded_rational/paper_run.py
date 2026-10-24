"""The stored sweep the paper's figure reads (results/paper_rate_utility/rateutil_20260922_144534), and how a
re-solved cell is compared with it."""
import io

import numpy as np
import pandas as pd

from bounded_optimal_confidence.paths import PAPER_RATE_UTILITY_DIR

PAPER_RUN_DIR = PAPER_RATE_UTILITY_DIR / "rateutil_20260922_144534"
KEY = ["world", "stimulus_config", "family", "beta3"]


def load_paper_curves(run_dir=PAPER_RUN_DIR):
    """curves.csv with the floats parsed bit-exactly, so a re-solved value can be compared with ==."""
    return pd.read_csv(run_dir / "curves.csv", float_precision="round_trip")


def as_written(rows, columns):
    """Rows from `rate_utility.cell`, taken through the same CSV round trip the sweep writes them with."""
    df = pd.DataFrame(rows).reindex(columns=columns)
    return pd.read_csv(io.StringIO(df.to_csv(index=False)), float_precision="round_trip")


def compare(new, stored):
    """Every column of `new` against the stored rows with the same (world, config, family, beta3).

    Returns a DataFrame with one row per new row and one boolean column per compared column (NaN matches NaN), plus
    the key columns. `result.drop(columns=KEY).all(axis=None)` is True when everything is reproduced exactly.
    """
    merged = new.merge(stored, on=KEY, how="left", suffixes=("", "_stored"), validate="one_to_one")
    out = merged[KEY].copy()
    for c in new.columns:
        if c in KEY:
            continue
        a, b = merged[c], merged[c + "_stored"]
        if a.dtype.kind in "fiub" and b.dtype.kind in "fiub":
            out[c] = (a.to_numpy() == b.to_numpy()) | (a.isna() & b.isna()).to_numpy()
        else:
            out[c] = (a.astype(object) == b.astype(object)) | (a.isna() & b.isna())
    return out


def worst_differences(new, stored):
    """{column: max |new - stored|} over the numeric columns that differ, for a report of a failed comparison."""
    merged = new.merge(stored, on=KEY, how="left", suffixes=("", "_stored"), validate="one_to_one")
    out = {}
    for c in new.columns:
        if c in KEY or new[c].dtype.kind not in "fiu" or merged[c + "_stored"].dtype.kind not in "fiu":
            continue
        d = np.abs(merged[c].to_numpy(dtype=float) - merged[c + "_stored"].to_numpy(dtype=float))
        d = d[~np.isnan(d)]
        if len(d) and d.max() > 0:
            out[c] = float(d.max())
    return out
