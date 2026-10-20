"""Attention as implicit structural inference: collapsed inference over a repertoire of attentional masks.

Notation: category c (the paper's latent state x), measurement x (the paper's observation w), structure phi.

A repertoire is N binary masks phi^(n) in {0,1}^K over the K categories. Each mask is a sparse prior over the
categories, and which one is used is inferred from the measurement:

    p(c|phi)      = phi_c / SUM_j phi_j
    p(x|phi)      = SUM_c p(c|phi) p(x|c)                      the structure's evidence
    p_beta(phi|x) ∝ p(phi) p(x|phi)^beta                        structure selection
    p(c|x,phi)    = p(c|phi) p(x|c) / p(x|phi)
    p(c|x)        = SUM_phi p_beta(phi|x) p(c|x,phi)            the attended posterior

beta tempers the evidence only, not the prior. beta = 1 is the exact Bayes mixture; beta = inf is hard attention,
the maximum-evidence structure alone (p(phi) then no longer matters); a finite fitted beta is soft attention.
p(phi) is flat over categories, p(phi) ∝ SUM_c phi_c. For a repertoire that partitions the categories (sym_spec,
asym_spec_l, asym_spec_r) that makes SUM_phi p(phi) p(c|phi) uniform, so beta = 1 is the standard Bayes posterior.

The likelihood p(x|c) enters only through ratios across c, so the flat-prior Bayes posterior (`bayes.bayes_posterior`)
can be passed in its place; that is what the models do.
"""
import numpy as np
from scipy.optimize import brentq

# The paper's four repertoires, one row per structure, columns = categories left to right. The row order is part of
# the arithmetic (the mixture sums over rows), so it is kept as in the paper's runs.
REPERTOIRES = {
    "sym_top2": np.array([[0.0, 1.0, 1.0], [1.0, 1.0, 0.0]]),       # drop one outer category
    "sym_spec": np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),       # the outer pair, or the middle alone
    "asym_spec_l": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]),    # the left category alone, or the other two
    "asym_spec_r": np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]),    # the right category alone, or the other two
}


def _check_masks(masks):
    masks = np.asarray(masks, dtype=float)
    if masks.ndim != 2 or not np.all((masks == 0) | (masks == 1)) or np.any(masks.sum(axis=1) == 0):
        raise ValueError("masks: an (N, K) array of 0/1 with at least one 1 in every row")
    return masks


def state_prior(masks):
    """p(c|phi) = phi_c / SUM_j phi_j, shape (N, K)."""
    return masks / masks.sum(axis=1, keepdims=True)


def structure_prior(masks):
    """The default p(phi) ∝ SUM_c phi_c: flat over categories. Shape (N,)."""
    n = masks.sum(axis=1)
    return n / n.sum()


def evidence(lik, masks):
    """p(x|phi) = SUM_c p(c|phi) p(x|c), shape (..., N)."""
    return lik @ state_prior(masks).T


def structure_posterior(lik, masks, beta=1.0, p_phi=None):
    """p_beta(phi|x) ∝ p(phi) p(x|phi)^beta, shape (..., N) for lik (..., K).

    beta = inf is the limit: p(phi) restricted to the maximum-evidence structures -- one-hot everywhere but exactly
    on a cut, where the tied structures share by their prior.
    """
    masks = _check_masks(masks)
    lik = np.asarray(lik, dtype=float)
    if lik.ndim < 1 or lik.shape[-1] != masks.shape[1]:
        raise ValueError(f"shapes: lik {lik.shape}, masks {masks.shape}")
    p_phi = structure_prior(masks) if p_phi is None else np.asarray(p_phi, dtype=float)
    if p_phi.shape != (len(masks),) or np.any(p_phi < 0) or not np.isclose(p_phi.sum(), 1.0):
        raise ValueError("p(phi): one non-negative weight per structure, summing to 1")
    ev = evidence(lik, masks)
    if np.any(np.where(p_phi > 0, ev, 0.0).max(axis=-1) <= 0):
        raise ValueError("p(x|phi) is zero for every structure with p(phi) > 0 at some x")
    if beta == 0:
        return np.broadcast_to(p_phi, ev.shape).copy()
    with np.errstate(divide="ignore"):
        log_prior, log_ev = np.log(p_phi), np.log(ev)
    if np.isinf(beta):
        log_ev = np.where(p_phi > 0, log_ev, -np.inf)
        logits = np.where(log_ev == log_ev.max(axis=-1, keepdims=True), log_prior, -np.inf)
    else:
        logits = log_prior + beta * log_ev
    p = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return p / p.sum(axis=-1, keepdims=True)


def state_given_structure(lik, masks):
    """p(c|x,phi) = p(c|phi) p(x|c) / p(x|phi), shape (..., N, K)."""
    joint = lik[..., None, :] * state_prior(masks)
    z = joint.sum(axis=-1, keepdims=True)
    if np.any(z <= 0):
        raise ValueError("p(x|phi) is zero at some (x, phi)")
    return joint / z


def posterior(lik, masks, beta=1.0, p_phi=None):
    """p(c|x) = SUM_phi p_beta(phi|x) p(c|x,phi), shape (..., K). Also returns p_beta(phi|x)."""
    weights = structure_posterior(lik, masks, beta, p_phi)
    # as state_given_structure, but a structure whose evidence underflowed to 0 is allowed where it has no weight
    joint = np.asarray(lik, dtype=float)[..., None, :] * state_prior(_check_masks(masks))
    z = joint.sum(axis=-1)
    if np.any((z <= 0) & (weights > 0)):
        raise ValueError("p(x|phi) is zero at some (x, phi) that carries weight")
    return np.einsum("...nx,...n->...x", joint / np.where(z > 0, z, 1.0)[..., None], weights), weights


def attend(post, repertoire, beta):
    """The attended posterior p(c|x) of a named repertoire at selection temperature beta (inf = hard attention),
    from the flat-prior posterior `post` (..., K)."""
    return posterior(post, REPERTOIRES[repertoire], beta)[0]


def top2_truncate(post):
    """Keep the two most probable categories and renormalise them, e.g. [0.5, 0.4, 0.1] -> [0.5556, 0.4444, 0].

    This is hard attention with the sym_top2 repertoire (`attend(post, "sym_top2", inf)`) everywhere off the tie of
    the two outer categories (checked in tests/test_attention.py). The Sym Top-2 hard model uses this form, as the
    paper's fits did.
    """
    top2_idx = np.argsort(post, axis=-1)[..., -2:]                 # (..., 2), ascending: [2nd, 1st]
    top2_p = np.take_along_axis(post, top2_idx, axis=-1)
    top2_p = top2_p / np.sum(top2_p, axis=-1, keepdims=True)
    out = np.zeros_like(post)
    np.put_along_axis(out, top2_idx, top2_p, axis=-1)
    return out


# ---------------------------------------------------------------- hard-attention boundaries

def gaussian_lik(x, means, sigma):
    """p(x|c) of Gaussian categories at 1-D measurements x, shape (len(x), K)."""
    m = np.asarray(means)[None, :]
    return np.exp(-0.5 * ((x[:, None] - m) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def cuts(masks, means, sigma, lo=-600.0, hi=600.0, step=0.25):
    """The beta = inf selection boundaries on a 1-D layout: the x in [lo, hi] where the maximum-evidence structure
    changes. They follow from the evidence, so they move with the layout and with sigma.

    Returns (cut points ascending, structure index in each of the len(cuts) + 1 intervals).
    """
    masks = _check_masks(masks)

    def log_ev(x):
        return np.log(evidence(gaussian_lik(np.atleast_1d(np.asarray(x, dtype=float)), means, sigma), masks))

    grid = np.arange(lo, hi + step, step)
    best = log_ev(grid).argmax(axis=1)
    points, regions = [], [int(best[0])]
    for i in np.flatnonzero(np.diff(best) != 0):
        a, b = best[i], best[i + 1]
        points.append(brentq(lambda x: np.subtract(*log_ev(x)[0, [a, b]]), grid[i], grid[i + 1], xtol=1e-12))
        regions.append(int(b))
    return np.array(points), regions
