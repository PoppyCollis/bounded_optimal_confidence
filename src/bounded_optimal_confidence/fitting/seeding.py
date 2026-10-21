"""The random streams of a fit (common random numbers).

Search stage: every grid cell of a subject draws from the SAME stream per config, `rank_rng(seed, subject, cfg)`, so
the profiled surface is smooth in the parameters rather than Monte-Carlo jittered: neighbouring cells differ only
because the parameters differ.

Clean stage: each shortlisted cell gets its own key, `clean_key(...)`, and two independent streams per config from
it -- `refit_rng` for the re-profile of the b-triple and lapse, `score_rng` for the clean NLL at those values -- so
the reported NLL is scored on draws the boundaries were not fitted to.

These keys reproduce the paper's fits exactly; changing any of them changes every number.
"""
import numpy as np


def rank_rng(seed, subject, cfg):
    return np.random.default_rng([seed, subject, cfg])


def alpha_word(log10_alpha):
    """A non-negative seed word for log10 alpha: round(100 log10 alpha), negatives mapped above 100000, inf 99999."""
    if not np.isfinite(log10_alpha):
        return 99999
    k = int(round(100 * log10_alpha))
    return k if k >= 0 else 100000 - k


def beta_word(log10_beta):
    """A non-negative seed word for log10 beta: round(1000 log10 beta), negatives mapped above 1000000, inf 9999999."""
    if not np.isfinite(log10_beta):
        return 9999999
    k = int(round(1000 * float(log10_beta)))
    return k if k >= 0 else 1000000 - k


def clean_key(seed, subject, sig_x_index, log10_alpha, log10_beta=None):
    """The clean stage's key of one cell. log10_beta is part of it for the soft-attention models only (None)."""
    key = [seed, subject, int(sig_x_index), alpha_word(log10_alpha)]
    return key if log10_beta is None else key + [beta_word(log10_beta)]


def refit_rng(key, block):
    """Stream for re-profiling the b-triple and lapse; `block` is the config's position in the pooled data."""
    return np.random.default_rng(key + [1, block])


def score_rng(key, block):
    """Stream for the clean NLL at the re-profiled values."""
    return np.random.default_rng(key + [2, block])
