"""Simulate a model's decisions and confidences on a subject's trials.

Per trial and replicate:

    x, y   = target + N(0, sig_x^2) in each dimension           noisy measurement
    p      = bayes_posterior(x, y; sig_p),  sig_p^2 = sig_s^2 + sig_x^2
    p      = the model's posterior: p (no attention), top-2 truncated (Sym Top-2 hard) or attended (`attention`)
    p~     ~ Dirichlet(alpha p)                                   decision noise (alpha = inf: p~ = p)
    d      = argmax p~
    conf   = readout(p~)

The random draws happen in that order (x noise, y noise, Dirichlet gammas), chunk by chunk, so a given generator
state always produces the same simulation. Keep that order: the fits rely on common random numbers.
"""
import numpy as np

from bounded_optimal_confidence.experiment import CATEGORY_X, CATEGORY_Y, SIG_S
from bounded_optimal_confidence.models.attention import attend, top2_truncate
from bounded_optimal_confidence.models.bayes import bayes_posterior
from bounded_optimal_confidence.models.readouts import READOUTS
from bounded_optimal_confidence.models.registry import get_model

CHUNK_REPS = 2000       # replicates simulated at once, to bound memory


def model_posterior(model, post, log10_beta=None):
    """The posterior `model` decides from, given the flat-prior Bayes posterior `post` (..., K)."""
    model = get_model(model)
    if model.attention == "none":
        return post
    if model.attention == "hard" and model.repertoire == "sym_top2":
        return top2_truncate(post)
    beta = 10.0 ** log10_beta if model.fits_beta and np.isfinite(log10_beta) else np.inf
    return attend(post, model.repertoire, beta)


def dirichlet_argmax(post, log10_alpha, rng):
    """(p~, d): p~ ~ Dirichlet(alpha p) via normalised gammas, d = argmax p~. A zero category stays zero."""
    if np.isinf(log10_alpha):
        return post, np.argmax(post, axis=-1)
    g = rng.gamma(post * 10.0 ** log10_alpha, scale=1.0)
    s = g.sum(axis=-1, keepdims=True)
    g = np.where(s > 0, g / np.where(s > 0, s, 1.0), post)      # all-underflow guard (tiny alpha)
    return g, np.argmax(g, axis=-1)


def simulate(model, cfg, trials, sig_x, log10_alpha, log10_beta, n_reps, rng):
    """(decisions, confidences), each (n_reps, n_trials): decisions 0-based, confidences in [0, 1].

    `log10_beta` is used by the soft-attention models only (inf = hard selection); pass None for the others.
    """
    model = get_model(model)
    readout = READOUTS[model.readout]
    tx, ty = trials.target_x, trials.target_y
    ux, uy = CATEGORY_X[cfg], CATEGORY_Y[cfg]
    sig_p = np.sqrt(SIG_S ** 2 + sig_x ** 2)
    ds, confs = [], []
    for start in range(0, n_reps, CHUNK_REPS):
        n = min(CHUNK_REPS, n_reps - start)
        x = tx[None] + rng.standard_normal((n, len(tx))) * sig_x
        y = ty[None] + rng.standard_normal((n, len(tx))) * sig_x
        post = model_posterior(model, bayes_posterior(x, y, ux, uy, sig_p), log10_beta)
        post, d = dirichlet_argmax(post, log10_alpha, rng)
        confs.append(readout(post))
        ds.append(d)
    return np.concatenate(ds), np.concatenate(confs)
