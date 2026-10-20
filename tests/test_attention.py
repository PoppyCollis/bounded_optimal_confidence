"""Collapsed inference over attentional masks: the identities the models rely on."""
import numpy as np
import pytest

from bounded_optimal_confidence.experiment import CATEGORY_X, CONFIGS, SIG_S
from bounded_optimal_confidence.models.attention import (REPERTOIRES, attend, cuts, evidence, gaussian_lik, posterior,
                                                         state_given_structure, state_prior, structure_prior,
                                                         top2_truncate)
from bounded_optimal_confidence.models.bayes import bayes_posterior

X = np.arange(-200.0, 201.0, 1.0)
LIK = gaussian_lik(X, CATEGORY_X[1], SIG_S)
STANDARD = LIK / LIK.sum(axis=1, keepdims=True)
BETAS = (0.0, 0.5, 1.0, 5.0, np.inf)
PARTITIONS = ("sym_spec", "asym_spec_l", "asym_spec_r")


def norm(q):
    return q / q.sum(axis=-1, keepdims=True)


@pytest.mark.parametrize("name", REPERTOIRES)
@pytest.mark.parametrize("p_phi", [None, np.array([0.3, 0.7])])
def test_beta_limits(name, p_phi):
    masks = REPERTOIRES[name]
    pp = structure_prior(masks) if p_phi is None else p_phi
    per, ev = state_given_structure(LIK, masks), evidence(LIK, masks)

    # beta = 1 is Bayes on the joint p(x, c, phi) = p(phi) p(c|phi) p(x|c)
    p_one, _ = posterior(LIK, masks, 1.0, p_phi)
    assert np.abs(p_one - norm(LIK * (pp @ state_prior(masks))[None, :])).max() < 1e-12

    # beta = 0 ignores the measurement in selection: the prior-weighted mix of the per-structure posteriors
    p_zero, w_zero = posterior(LIK, masks, 0.0, p_phi)
    assert np.allclose(w_zero, pp) and np.allclose(p_zero, np.einsum("wnx,n->wx", per, pp))

    # beta = inf is the maximum-evidence structure's posterior, whatever p(phi) is (off the cuts) ...
    p_inf, _ = posterior(LIK, masks, np.inf, p_phi)
    off_cut = ev[:, 0] != ev[:, 1]
    assert np.allclose(p_inf[off_cut], per[np.arange(len(X)), ev.argmax(axis=1)][off_cut])
    # ... and a large finite beta approaches it, away from the cuts
    sharp = np.abs(np.log(ev[:, 0] / ev[:, 1])) > 0.1
    assert np.abs(posterior(LIK, masks, 1e4, p_phi)[0] - p_inf)[sharp].max() < 1e-9

    for beta in BETAS:
        p, w = posterior(LIK, masks, beta, p_phi)
        assert np.allclose(p.sum(axis=1), 1.0) and np.allclose(w.sum(axis=1), 1.0)


@pytest.mark.parametrize("name", PARTITIONS)
def test_partition_at_beta_one_is_bayes(name):
    assert np.all(REPERTOIRES[name].sum(axis=0) == 1)
    assert np.abs(posterior(LIK, REPERTOIRES[name], 1.0)[0] - STANDARD).max() < 1e-12


@pytest.mark.parametrize("name", REPERTOIRES)
@pytest.mark.parametrize("beta", BETAS)
def test_shape_and_rescaling_invariance(name, beta):
    """Any leading shape, and the flat-prior posterior in place of the likelihood, give the same answer."""
    flat, flat_w = posterior(LIK, REPERTOIRES[name], beta)
    p, w = posterior(STANDARD[:300].reshape(3, 100, 3), REPERTOIRES[name], beta)
    assert np.abs(p - flat[:300].reshape(3, 100, 3)).max() < 1e-12
    assert np.abs(w - flat_w[:300].reshape(3, 100, 2)).max() < 1e-12


def test_cuts_on_config_1():
    # asym_spec_l: N(x; -d) = [N(x; 0) + N(x; d)] / 2  <=>  t^2 + e^(d^2 / 2 sigma^2) t - 2 = 0,  t = e^(d x / sigma^2)
    d, e = 96.0, np.exp(96.0 ** 2 / (2 * SIG_S ** 2))
    x_star = SIG_S ** 2 / d * np.log((-e + np.sqrt(e * e + 8)) / 2)
    means = CATEGORY_X[1]
    pts, regions = cuts(REPERTOIRES["asym_spec_l"], means, SIG_S)
    assert len(pts) == 1 and abs(pts[0] - x_star) < 1e-9 and regions == [0, 1]
    pts, regions = cuts(REPERTOIRES["asym_spec_r"], means, SIG_S)
    assert len(pts) == 1 and abs(pts[0] + x_star) < 1e-9 and regions == [1, 0]
    pts, regions = cuts(REPERTOIRES["sym_top2"], means, SIG_S)
    assert len(pts) == 1 and abs(pts[0]) < 1e-9 and regions == [1, 0]
    pts, regions = cuts(REPERTOIRES["sym_spec"], means, SIG_S)
    assert len(pts) == 2 and abs(pts[0] + pts[1]) < 1e-9 and regions == [0, 1, 0]


def test_zero_evidence_structure():
    """A structure with zero evidence is dropped where it carries no weight, and refused where it does."""
    dead = np.array([[0.0, 0.2, 0.8]])
    assert np.allclose(posterior(dead, REPERTOIRES["asym_spec_l"], np.inf)[0], dead)
    assert np.allclose(posterior(dead, REPERTOIRES["asym_spec_l"], 2.0)[0], dead)
    with pytest.raises(ValueError):
        posterior(dead, REPERTOIRES["asym_spec_l"], 0.0)


# The model-derived (hard) attention is a hand-placed mask switch at the derived cuts: written out here independently.
HAND_MASKS = {"asym_spec_l": np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]]),
              "asym_spec_r": np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])}
SYM_MASKS = np.array([[0.0, 1.0, 0.0], [0.5, 0.0, 0.5]])         # inside the two cuts, outside them


def hand_cut(post, x, cut, pcp):
    return norm(post * np.where((x < cut)[..., None], pcp[0], pcp[1]))


def hand_band(post, x, lo, hi, pcp):
    return norm(post * np.where(((x >= lo) & (x < hi))[..., None], pcp[0], pcp[1]))


@pytest.mark.parametrize("cfg", CONFIGS)
@pytest.mark.parametrize("sig_p", [64.0, 150.0])
def test_hard_attention_is_a_cut_at_the_derived_boundaries(cfg, sig_p):
    x = np.arange(-300.0, 301.0, 0.5)
    ux = CATEGORY_X[cfg]
    post = bayes_posterior(x, np.zeros_like(x), ux, 0.0, sig_p)
    for name in PARTITIONS:
        pts, _ = cuts(REPERTOIRES[name], ux, sig_p, lo=-2000.0, hi=2000.0)
        off = np.all(np.abs(x[:, None] - pts[None, :]) > 1e-6, axis=1)
        hard = attend(post, name, np.inf)
        ref = hand_band(post, x, pts[0], pts[1], SYM_MASKS) if name == "sym_spec" else \
            hand_cut(post, x, pts[0], HAND_MASKS[name])
        assert np.abs(hard - ref)[off].max() < 1e-12
        # soft attention: beta = 1 on a partition is the Bayes posterior
        assert np.abs(attend(post, name, 1.0) - post).max() < 1e-12
    # Sym Top-2: hard attention is top-2 truncation, off the tie of the outer pair at x = 0
    off0 = x != 0
    assert np.abs(attend(post, "sym_top2", np.inf) - top2_truncate(post))[off0].max() < 1e-12


def test_top2_truncate():
    q = top2_truncate(np.array([[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]]))
    assert np.allclose(q, [[5 / 9, 4 / 9, 0.0], [0.0, 1 / 3, 2 / 3]])
