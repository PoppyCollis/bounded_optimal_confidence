"""Bootstrapped confidence interval of a statistic of a vector (percentile method).

The entries of the vector are resampled with replacement, the statistic is recomputed on every resample, and the
interval is the central `level` % of those values. With one entry per subject this is the across-subject
uncertainty of the statistic.

    lo, hi = bootstrap_ci(x)                    # 95% CI of the mean
    lo, hi = bootstrap_ci(x, stat=np.sum)       # 95% CI of the sum
"""
import numpy as np


def bootstrap_ci(x, stat=np.mean, n_boot=10_000, level=95, seed=0):
    """(lo, hi) percentile-bootstrap confidence interval of stat(x).

    `stat` is called as stat(resamples, axis=1) on an (n_boot, len(x)) array, so np.mean / np.sum / np.median work.
    The seed is fixed, so two vectors of the same length are resampled with the same indices (paired resamples).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) == 0:
        raise ValueError("x must be a non-empty 1-D vector")
    idx = np.random.default_rng(seed).integers(0, len(x), size=(n_boot, len(x)))
    tail = (100 - level) / 2
    lo, hi = np.percentile(stat(x[idx], axis=1), [tail, 100 - tail])
    return float(lo), float(hi)
