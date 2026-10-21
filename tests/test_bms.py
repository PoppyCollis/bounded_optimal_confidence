"""Bayesian model selection: identical to the groupBMC reference implementation, sane on synthetic evidence, and the
paper's numbers."""
import numpy as np
import pytest
from scipy.stats import dirichlet

from bounded_optimal_confidence.runs import load_fits
from bounded_optimal_confidence.stats.bms import (MAX_ITER, TOLERANCE, bms, exceedance_probability, family_bms,
                                                   null_log_evidence, omnibus_risk, variational_fit)
from bounded_optimal_confidence.stats.tables import aic_table, bms_table, family_bms_tables, log_evidence

groupBMC = pytest.importorskip("groupBMC.groupBMC")

FITS = load_fits(paper=True)
L_PAPER = log_evidence(aic_table(FITS))                         # 10 models x 13 subjects


def random_evidence(K, N, seed, spread=3.0):
    rng = np.random.default_rng(seed)
    return spread * rng.standard_normal((K, N))


@pytest.mark.parametrize("L", [L_PAPER, random_evidence(4, 20, 1), random_evidence(3, 7, 2, spread=0.5)])
def test_variational_fit_matches_groupbmc(L):
    """The same counts and free energy as the package under the flat prior alpha_0 = 1."""
    K = L.shape[0]
    g = groupBMC.GroupBMC(L, α_0=np.ones(K), max_iter=MAX_ITER, tolerance=TOLERANCE)
    alpha, z, F1 = variational_fit(L, np.ones(K))
    assert np.array_equal(alpha, g.α.ravel())
    assert np.array_equal(z, g.z)
    assert F1 == g.F1()


def test_exceedance_probability_matches_groupbmc():
    alpha = np.array([3.5, 1.2, 7.9, 1.0])
    assert np.array_equal(exceedance_probability(alpha), groupBMC.exceedance_probability(dirichlet(alpha)))


def test_bor_and_pxp_equal_the_package_under_its_default_prior():
    """groupBMC's own BOR is right only for alpha_0 = 1 / K. There, our exact null reproduces it."""
    L = random_evidence(4, 20, 3)
    K = L.shape[0]
    g = groupBMC.GroupBMC(L, max_iter=MAX_ITER, tolerance=TOLERANCE)      # default prior 1 / K per model
    alpha, _z, F1 = variational_fit(L, np.full(K, 1 / K))
    bor = omnibus_risk(L, np.full(K, 1 / K), F1)
    res = g.get_result()
    assert bor == pytest.approx(1 / (1 + np.exp(g.F1() - g.F0())), rel=1e-12)
    assert np.allclose(exceedance_probability(alpha) * (1 - bor) + bor / K, res.protected_exceedance_probability,
                       atol=1e-15)


def test_exceedance_probability_sums_to_one_and_matches_sampling():
    alpha = np.array([6.0, 4.0, 1.5, 1.5, 1.0])
    xp = exceedance_probability(alpha)
    assert xp.sum() == pytest.approx(1.0)
    draws = dirichlet(alpha).rvs(200_000, random_state=0)
    mc = np.bincount(draws.argmax(axis=1), minlength=len(alpha)) / len(draws)
    assert np.allclose(xp, mc, atol=0.005)


def test_null_log_evidence_is_the_mixture_evidence():
    L = np.log(np.array([[0.5, 0.2], [0.5, 0.8]]))
    assert null_log_evidence(L, [0.5, 0.5]) == pytest.approx(np.log(0.5) + np.log(0.5))


def test_one_dominant_model_is_found():
    L = np.zeros((3, 30))
    L[1] = 20.0                                                 # every subject strongly favours model 2
    r = bms(L)
    assert np.allclose(r.alpha, [1, 31, 1])                     # the prior count plus the attributed subjects
    assert r.frequency.argmax() == 1 and r.frequency[1] > 0.9
    assert r.pxp[1] > 0.99 and r.bor < 1e-6


def test_indistinguishable_models_give_a_high_omnibus_risk():
    r = bms(np.zeros((3, 15)))
    assert np.allclose(r.frequency, 1 / 3)
    assert r.bor > 0.8                                          # 0.88: the null is simpler, so it is favoured
    assert np.allclose(r.pxp, 1 / 3, atol=0.01)


def test_family_counts_are_the_summed_member_counts():
    L = random_evidence(3, 10, 4)
    sizes = [1, 2]
    alpha, _z, _F = variational_fit(L, [1.0, 0.5, 0.5])         # the Penny prior
    r = family_bms(L, sizes, "penny")
    assert np.allclose(r.alpha, [alpha[0], alpha[1] + alpha[2]])
    assert r.frequency.sum() == pytest.approx(1.0) and r.xp.sum() == pytest.approx(1.0)


def test_paper_model_level_bms():
    """The paper's numbers: the differences are real (BOR 0.0015), soft Sym Top-2 and Diff share the probability."""
    tab, bor = bms_table(FITS)
    pxp = tab.set_index("model").pxp
    assert bor == pytest.approx(0.0015212, rel=1e-4)
    assert pxp["sym_top2_soft"] == pytest.approx(0.51288, abs=1e-4)
    assert pxp["diff"] == pytest.approx(0.47109, abs=1e-4)
    assert pxp["sym_top2_hard"] == pytest.approx(0.00546, abs=1e-4)
    assert (pxp.drop(["diff", "sym_top2_soft", "sym_top2_hard"]) < 0.002).all()
    assert tab.set_index("model").best[["diff", "sym_top2_soft", "sym_top2_hard"]].tolist() == [6, 5, 2]


def test_paper_family_bms():
    """Diff vs the Sym Top-2 family: no evidence either way (BOR > 0.6, PXP near 1 / 2)."""
    (_t1, m, bor_m), (_t2, penny, bor_p), (_t3, default, bor_d) = family_bms_tables(FITS)
    assert bor_m == pytest.approx(0.6222, abs=1e-3)
    assert bor_p == pytest.approx(0.6783, abs=1e-3)
    assert bor_d == pytest.approx(0.7158, abs=1e-3)
    assert np.allclose(m.pxp, [0.3859, 0.2094, 0.4047], atol=1e-3)
    assert np.allclose(penny.pxp, [0.4882, 0.5118], atol=1e-3)
    assert np.allclose(default.pxp, [0.4859, 0.5141], atol=1e-3)
    assert penny.best.tolist() == [6, 7]
