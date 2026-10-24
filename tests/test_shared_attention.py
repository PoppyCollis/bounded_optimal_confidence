"""The shared attention optimum: the grid, the samplers, a small solve, the figure's reader, and the paper's sweep
reproduced (one job, under -m slow)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from bounded_optimal_confidence.shared_attention import optimise_attention_discrete as oa
from bounded_optimal_confidence.shared_attention import paper_run

SWEEP = paper_run.load_paper_sweep()


def _configure_paper():
    oa.configure(**paper_run.PAPER_CFG)


def test_grid_sizes_match_the_stored_sweep():
    _configure_paper()
    for b, n in SWEEP[["B", "n_grid"]].drop_duplicates().itertuples(index=False):
        assert oa.n_grid_for(float(b)) == n
        oa.configure(B=float(b), N_GRID=int(n))
        assert len(oa.W) == n and oa.W[0] == -oa.W[-1] and oa.W[n // 2] == 0.0
        # the step the figure recovers from (B, n_grid), optimise_attention.make_grid's own
        assert np.isclose(oa.DW, 2 * (b + 5 * 64.0) / (n - 1))
        assert oa.DW <= oa.SWEEP_STEP * 64.0 + 1e-12


def test_edge_priors_and_supports():
    assert np.array_equal(oa.edge_prior(0, 0.25), [0.75, 0.25, 0.0])
    assert np.array_equal(oa.edge_prior(2, 1.0), [0.0, 0.0, 1.0])
    assert oa.support_name(np.array([0.5, 0.5, 0.0])) == "{left, middle}"
    assert oa.support_name(np.array([0.0, 0.0, 1.0])) == "{right}"


def test_recentred_sampler():
    rng = np.random.default_rng(0)
    mu = oa.sample_recentred(1000, 250.0, rng)
    assert mu.shape == (1000, 3)
    assert np.allclose(mu[:, 0], -mu[:, 2]) and (np.diff(mu, axis=1) >= 0).all()
    assert (np.abs(mu[:, 1]) <= mu[:, 2]).all() and (mu[:, 2] <= 250).all()
    # the same seed draws the same worlds (what makes the sweep reproducible)
    assert np.array_equal(mu, oa.sample_recentred(1000, 250.0, np.random.default_rng(0)))


def test_small_solve_orders_the_codes():
    oa.configure(**{**paper_run.PAPER_CFG, "N_WORLDS": 200, "N_GRID": 201, "N_T": 21, "B": 250.0, "SEED": 0})
    r = oa.solve()
    for pi in (r["pi1"], r["pi2"]):
        assert np.isclose(pi.sum(), 1) and (pi >= 0).all() and (pi == 0).sum() >= 1   # sparse: on an edge
    assert r["pi1"] @ [-1, 0, 1] <= r["pi2"] @ [-1, 0, 1]                              # phi_1 is the left one
    fit, held = r["fit"], r["held"]
    assert fit["delta"] >= fit["opt"] >= fit["second"] - 1e-12     # the bound, the optimum, the runner-up (fit worlds)
    assert held["delta"] >= held["opt"]                            # the bound holds on any worlds
    _configure_paper()


def _load_figure_module():
    path = Path(__file__).resolve().parents[1] / "paper_plots" / "fig_pxgphi_vs_B.py"
    spec = importlib.util.spec_from_file_location("fig_pxgphi_vs_B", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_figure_reader_on_the_paper_sweep():
    fig = _load_figure_module()
    prior, sigma, x, pi, cut, xs = fig.load(str(paper_run.PAPER_RUN_DIR))
    assert prior == "recentred" and sigma == 64.0
    assert len(x) == 30 and len(xs) == 6 and pi.shape == (30, 2, 3)
    assert np.allclose(pi.sum(2), 1)
    assert (np.abs(cut) <= oa.SWEEP_STEP + 1e-12).all()        # every cut within one grid step of 0
    assert "_discrete_" in paper_run.PAPER_RUN_DIR.name


@pytest.mark.slow
def test_one_job_of_the_paper_sweep_is_reproduced():
    """(B 64, seed 0), the cheapest job: every non-timing column equal to the stored row."""
    row = oa._sweep_job((paper_run.PAPER_CFG, 64.0, 0))
    new = paper_run.as_written([row])
    result = paper_run.compare(new, SWEEP)
    assert result.drop(columns=paper_run.KEY).all(axis=None), paper_run.worst_differences(new, SWEEP)
