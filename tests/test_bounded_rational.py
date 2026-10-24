"""The bounded-rational hierarchy: the worlds, the identities the sweep checks on itself, and the paper's stored
sweep reproduced (the solved family under -m slow)."""
import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from bounded_optimal_confidence.bounded_rational import candidates, paper_run, rate_utility, seeds, transfer
from bounded_optimal_confidence.bounded_rational import world as gworld
from bounded_optimal_confidence.bounded_rational.info_terms import rate_utility as eu_and_rate
from bounded_optimal_confidence.bounded_rational.rate_distortion import onevar_ba
from bounded_optimal_confidence.bounded_rational.rate_utility import BETA2, cell, score_pathway, world_kw_for

CURVES = paper_run.load_paper_curves()
Q = float(CURVES["q"].iloc[0])
HEAD = (float(CURVES["beta1"].iloc[0]), BETA2)


@pytest.mark.parametrize("world, cfg, lo, n, d1", [
    ("raw", 1, -160, 1000, 48), ("raw", 2, -192, 1000, 64), ("raw", 3, -160, 1000, 77.5), ("raw", 4, -160, 1000, 18.5),
    ("norm", 1, -480, 2998, 48), ("norm", 2, -512, 3198, 64), ("norm", 3, -480, 2998, 77.5), ("norm", 4, -480, 2998, 18.5)])
def test_worlds_match_the_stored_config_log(world, cfg, lo, n, d1):
    wd, U, pw = gworld.build(cfg, **world_kw_for(world))
    assert wd["w_vals"][0] == lo and wd["w_vals"][-1] == -lo and len(wd["w_vals"]) == n
    assert wd["exclusion_cut"] == 0 and wd["boundary"] == d1
    assert wd["mirror_symmetric"] == (cfg in (1, 2))
    assert U.shape == (3, n) and pw.shape == (n,) and abs(pw.sum() - 1) < 1e-12


def test_beta3_axis():
    axis = rate_utility.beta3_axis()
    assert len(axis) == 80 and axis[0] == 0.05 and axis[-1] == 100.0
    assert 1.0 in axis and np.all(np.diff(axis) > 0)
    assert set(CURVES["beta3"]) == set(axis)


def test_rate_distortion_at_beta_1_is_bayes_in_the_normalised_world():
    """At beta3 = 1 single-variable BA on the normalised world gives p(a) = 1/3 and the flat-prior posterior."""
    for cfg in (1, 2, 3, 4):
        wd, U, pw = gworld.build(cfg, **world_kw_for("norm"))
        pa, pagw, _n = onevar_ba(1.0, U, pw)
        q = np.exp(U)
        assert np.abs(pa - 1 / 3).max() < 1e-9
        assert np.abs(pagw - q / q.sum(axis=0, keepdims=True)).max() < 1e-9


def test_top2_at_beta3_1_is_the_hard_top2_posterior():
    for world in ("raw", "norm"):
        res = rate_utility.check_top2_is_bayes((1, 2, 3, 4), world_kw_for(world))
        assert res["ok"], res


def test_masked_scoring_equals_transfer_score():
    for world in ("raw", "norm"):
        res = rate_utility.check_lean((1, 2, 3, 4), world_kw_for(world), HEAD, Q)
        assert res["ok"] and res["cells compared"] == 12, res


def test_hard_router_identity():
    """On the pinned top-2 family I(W;A) = I(A;O) + I(W;A|O), I(A;O|W) = 0 and I(W;O) = H(p(o))."""
    for world in ("raw", "norm"):
        kw = world_kw_for(world)
        for cfg in (1, 3):
            wd, U, pw = gworld.build(cfg, **kw)
            pogw, pago = candidates.top2_pathway(wd, Q)
            po = pogw @ pw
            for b3 in (0.3, 1.0, 20.0):
                r = score_pathway(pogw, pago, wd["w_vals"], cfg, (HEAD[0], HEAD[1], b3), kw)
                assert abs(r["I_AO"] + r["I_WAgO"] - r["I_WA"]) < 1e-9
                assert abs(r["I_AOgW"]) < 1e-9
                assert abs(r["I_WO"] + (po * np.log(po)).sum()) < 1e-9


def _stored(world, cfg, family, beta3):
    s = CURVES[(CURVES.world == world) & (CURVES.stimulus_config == cfg) & (CURVES.family == family)
               & (CURVES.beta3 == beta3)]
    assert len(s) == 1
    return s.iloc[0]


@pytest.mark.parametrize("beta3", [0.05, 1.0, 100.0])
def test_rd_and_top2_rows_of_the_paper_run_are_reproduced(beta3):
    """The two families that need no full solve: every stored number comes back exactly (config 1, norm world)."""
    kw = world_kw_for("norm")
    wd, U, pw = gworld.build(1, **kw)
    pogw, pago = candidates.top2_pathway(wd, Q)
    got = score_pathway(pogw, pago, wd["w_vals"], 1, (HEAD[0], HEAD[1], beta3), kw)
    want = _stored("norm", 1, "top2", beta3)
    for k in ("E_U", "J", "I_WO", "I_AO", "I_WAgO", "I_WA", "I_AOgW", "n_actions_used"):
        assert got[k] == want[k], (k, got[k], want[k])
    assert got["I_WAgO"] / beta3 == want["cost_cond"] and got["I_WA"] / beta3 == want["cost_total"]

    _pa, pagw, _n = onevar_ba(beta3, U, pw)
    eu, i_wa = eu_and_rate(pagw, U, pw)
    want = _stored("norm", 1, "rd", beta3)
    assert eu == want["E_U"] and i_wa == want["I_WA"] and eu - i_wa / beta3 == want["J"]


@pytest.mark.slow
@pytest.mark.parametrize("world, cfg, beta3", [("raw", 3, 1.0), ("norm", 1, 100.0)])
def test_cell_of_the_paper_run_is_reproduced(world, cfg, beta3):
    """A whole cell re-solved (three seeds, full BA, the collapsed reference, the RD frontier): identical rows."""
    with threadpool_limits(1):
        rows = cell((world_kw_for(world), cfg, beta3, Q, HEAD, True))
    new = paper_run.as_written(rows, list(CURVES.columns))
    result = paper_run.compare(new, CURVES)
    assert result.drop(columns=paper_run.KEY).all(axis=None), paper_run.worst_differences(new, CURVES)


def test_state_from_matches_the_solver_completion_at_beta2_inf():
    """`seeds.state_from` on the top-2 pathway and `transfer.score` agree with `score_pathway` (the lean path)."""
    kw = world_kw_for("raw")
    wd, U, pw = gworld.build(2, **kw)
    pogw, pago = candidates.top2_pathway(wd, Q)
    st = seeds.state_from(pogw, pago, BETA2, 2.0, U, pw)
    assert np.isfinite(st[4]).all()
    want = transfer.score(pogw, pago, wd["w_vals"], 2, (HEAD[0], BETA2, 2.0), kw)
    got = score_pathway(pogw, pago, wd["w_vals"], 2, (HEAD[0], BETA2, 2.0), kw)
    assert all(got[k] == want[k] for k in ("E_U", "J", "I_WA", "I_WAgO"))
