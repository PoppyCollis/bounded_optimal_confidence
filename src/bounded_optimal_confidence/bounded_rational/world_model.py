"""The world the hierarchy is solved in: U(a, w) = log N(w; mu_a, sigma) on a w grid, and p(w).

`_world_model` is the raw world of ITBR_confidence/beta_sweep/objective_task.py (raw log-likelihoods on a grid EPSILON
past the outer means, p(w) uniform or the mixture); `normalised_world`, `raw_world` and `build` are
ITBR_confidence/beta_sweep/norm_world.py. In the normalised world U is log q(w|a) with q normalised over the grid and
p(w) is the equal-weight mixture, which is the world in which 1-step rate distortion at beta = 1 is the Bayes posterior.

Deviation from the original: `_world_model` takes `mu_as` as a required argument (its default read a module global
that every caller here overrides).
"""
import numpy as np
from scipy.special import logsumexp

from bounded_optimal_confidence.bounded_rational import settings


def _world_model(mu_as, sigma=None, epsilon=None, n_samples=None, pw_prior=None):
    """(U_pre, pw) for the swept task; each argument but `mu_as` defaults to `settings`.

    `pw_prior` picks the stimulus prior on the grid (settings.PW_PRIORS): "uniform" is flat,
    "mixture" is sum_a p(a) N(w; mu_a, sigma) with p(a) flat, renormalised over the grid. U_pre is
    the same either way.

    U(a, w) = log N(w; mu_a, sigma) on a uniform grid of world states. Must stay identical to
    `simulate_br.br_utility` and to `probe_convergence._world_model`, which reconstruct and probe
    the solutions this produces -- see the note at the top of simulate_br.py.

    The arguments exist for callers that vary the world model *within* one process -- notably
    profile_readout_ridge, which walks an EPSILON ladder across four stimulus configs. Overriding
    beta_sweep.config for that would mean one worker pool per (config, epsilon), since the
    override reaches spawned children only through the environment they were created with; passing
    the world model with the task avoids that entirely and keeps it visible at the call site.
    """
    mu_as = np.asarray(mu_as, dtype=float)
    sigma = settings.SIGMA if sigma is None else sigma
    epsilon = settings.EPSILON if epsilon is None else epsilon
    n_samples = settings.N_SAMPLES if n_samples is None else n_samples
    pw_prior = settings.PW_PRIOR if pw_prior is None else pw_prior

    sigma_as = np.ones(len(mu_as)) * sigma
    w_vals = np.linspace(mu_as[0] - epsilon, mu_as[-1] + epsilon, n_samples)
    U_pre = np.array([
        [-0.5 * np.log(2 * np.pi * s ** 2) - 0.5 * ((v - m) ** 2) / s ** 2 for v in w_vals]
        for m, s in zip(mu_as, sigma_as)
    ])
    if pw_prior == "uniform":
        pw = np.ones_like(w_vals) / n_samples
    elif pw_prior == "mixture":
        # U_pre is exactly log N(w; mu_a, sigma), so this is sum_a (1/|A|) N(w; mu_a, sigma). The
        # grid is evenly spaced, so normalising the densities gives the probability masses.
        pw = np.exp(U_pre).mean(axis=0)
        pw /= pw.sum()
    else:
        raise ValueError(f"pw_prior must be one of {settings.PW_PRIORS}, got {pw_prior!r}")
    return U_pre, pw


SIG_S = 64.0            # stimulus width; human_data_model_fitting.config.STI_PAR["sig_s"] in the original


EPS_SD = 6.0            # grid half-width past the outer means, in units of sig_p


GRID_STEP = (96.0 - -96.0 + 2 * 64.0) / 999.0


def normalised_world(mu_as, sig_x=0.0, sig_s=SIG_S, eps_sd=EPS_SD, grid_step=GRID_STEP):
    """The grid-normalised, mixture-prior world for these category means.

    Returns a dict with the two things every solver here needs -- `U_pre` (card_a, n) and `pw` (n) --
    plus the grid and the numbers that describe it: `w_vals`, `mu_as`, `sig_p`, `sig_x`, `epsilon`,
    `n_samples`, `grid_step`, and `offset`.

    `offset` is the per-action constant relating U_pre to the bare kernel -(w - mu_a)^2 / (2 sig_p^2),
    for a caller that evaluates the policy OFF the grid (fit_sampled_decision.rd_pagx does this);
    nothing in the on-grid solve uses it.
    """
    mu_as = np.asarray(mu_as, dtype=float)
    sig_p = float(np.sqrt(sig_s ** 2 + sig_x ** 2))
    epsilon = float(eps_sd * sig_p)
    n_samples = int(np.ceil((mu_as[-1] - mu_as[0] + 2 * epsilon) / grid_step)) + 1
    # pw_prior is irrelevant here (the uniform one is discarded below), but naming it keeps the call
    # explicit about not relying on settings.PW_PRIOR, which travels in the environment.
    U_raw, _pw_uniform = _world_model(mu_as=mu_as, sigma=sig_p, epsilon=epsilon,
                                      n_samples=n_samples, pw_prior="uniform")
    log_norm = logsumexp(U_raw, axis=1)                   # log sum_w N(w; mu_a, sig_p)
    U_pre = U_raw - log_norm[:, None]                     # log q, sum_w q(w|a) = 1
    pw = np.exp(U_pre).mean(axis=0)
    offset = -0.5 * np.log(2 * np.pi * sig_p ** 2) - log_norm
    return {"w_vals": np.linspace(mu_as[0] - epsilon, mu_as[-1] + epsilon, n_samples),
            "U_pre": U_pre, "pw": pw / pw.sum(), "mu_as": mu_as, "sig_p": sig_p,
            "sig_x": float(sig_x), "sig_s": float(sig_s), "epsilon": epsilon,
            "n_samples": n_samples, "grid_step": float(grid_step),
            "offset": offset - offset.max()}


def raw_world(mu_as, epsilon=64.0, sigma=SIG_S, n_samples=1000, pw_prior="uniform"):
    """The OLD world -- raw log N, uniform p(w), narrow grid -- in the same dict shape.

    Here only so that a run made before this module existed can be reproduced, and so that a
    comparison between the two worlds is one argument rather than two code paths.
    """
    mu_as = np.asarray(mu_as, dtype=float)
    U_pre, pw = _world_model(mu_as=mu_as, sigma=sigma, epsilon=epsilon, n_samples=n_samples,
                             pw_prior=pw_prior)
    return {"w_vals": np.linspace(mu_as[0] - epsilon, mu_as[-1] + epsilon, n_samples),
            "U_pre": U_pre, "pw": pw, "mu_as": mu_as, "sig_p": float(sigma), "sig_x": 0.0,
            "sig_s": float(sigma), "epsilon": float(epsilon), "n_samples": int(n_samples),
            "grid_step": float((mu_as[-1] - mu_as[0] + 2 * epsilon) / (n_samples - 1)),
            "offset": np.zeros(len(mu_as))}


WORLDS = ("norm_mixture", "raw_uniform")


def build(world, mu_as, sig_x=0.0, **kw):
    """`normalised_world` or `raw_world` by name, for a CLI that offers both."""
    if world == "norm_mixture":
        return normalised_world(mu_as, sig_x=sig_x, **kw)
    if world == "raw_uniform":
        if sig_x:
            raise ValueError("raw_uniform has no sig_x; it is the pre-correction world")
        return raw_world(mu_as, **kw)
    raise ValueError(f"world must be one of {WORLDS}, got {world!r}")
