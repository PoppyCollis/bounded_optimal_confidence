"""The standard Bayesian decision model: the category posterior at a noisy measurement.

The observer measures the target at (x, y) with Gaussian noise of standard deviation sig_x and knows the category
layout, so the measurement of a category-c target is distributed N((ux_c, uy_c), sig_p^2) with
sig_p^2 = sig_s^2 + sig_x^2. With a flat category prior the posterior is

    p(c | x, y) ∝ exp(-[(x - ux_c)^2 + (y - uy_c)^2] / (2 sig_p^2))

Every model in this package starts from this posterior: Diff and Ent use it as it is, and the attention models
reshape it (`attention`) before the decision noise.
"""
import numpy as np


def bayes_posterior(x, y, ux, uy, sig_p):
    """Normalised category posterior at measurements (x, y), shape (..., n_categories).

    Computed in log space with the row maximum subtracted, so far-out measurements (large sig_x) do not underflow.
    """
    d2 = (x[..., None] - ux) ** 2 + (y[..., None] - uy) ** 2
    logp = -d2 / (2 * sig_p ** 2)
    logp -= logp.max(axis=-1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(axis=-1, keepdims=True)
