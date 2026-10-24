"""The explanatory paper figures: the 1D readout traces, the config constants, the delta-AIC summary.

The paper_plots scripts are loaded from their files (they are run by path, not imported as a package)."""
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from bounded_optimal_confidence.experiment import CATEGORY_X, SIG_S
from bounded_optimal_confidence.plotting.readout_traces import (W, likelihoods, max_prob, neg_entropy, posterior,
                                                                prob_difference, rescale, top2_neg_entropy)

PAPER_PLOTS = Path(__file__).resolve().parents[1] / "paper_plots"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, PAPER_PLOTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- readout_traces -------------------------------------------------------------------------------------------------

def test_posterior_is_normalised_and_mirror_symmetric():
    post = posterior(W)
    assert np.allclose(post.sum(1), 1)
    assert np.allclose(post, post[::-1, ::-1])                  # mu = (-1, 1): w -> -w swaps the two options
    assert post[np.argmin(np.abs(W))].tolist() == pytest.approx([0.5, 0.5])


def test_neg_entropy_bounds():
    post = posterior(W)
    assert (neg_entropy(post) <= 1e-12).all()
    assert neg_entropy(np.array([[1.0, 0.0]]))[0] == 0
    assert neg_entropy(np.array([[0.5, 0.5]]))[0] == pytest.approx(-np.log(2))


def test_two_option_identities():
    post = posterior(W)
    assert np.allclose(prob_difference(post), 2 * max_prob(post) - 1)
    assert np.allclose(top2_neg_entropy(post), neg_entropy(post))


def test_three_option_top2_keeps_the_two_most_likely():
    post = posterior(W, (-2.0, 0.0, 2.0), 1.5, (1 / 3,) * 3)
    top2 = np.sort(post, 1)[:, -2:]
    assert np.allclose(top2_neg_entropy(post), neg_entropy(top2 / top2.sum(1, keepdims=True)))
    assert (top2_neg_entropy(post) >= -np.log(2) - 1e-12).all()


def test_rescale_by_theory_maps_the_bounds_to_0_and_1():
    y = np.array([-np.log(2), -0.5, 0.0])
    assert rescale(y, (-np.log(2), 0), "theory").tolist() == pytest.approx([0, 1 - 0.5 / np.log(2), 1])
    assert rescale(y, (-np.log(2), 0), "data").tolist() == pytest.approx([0, 1 - 0.5 / np.log(2), 1])
    assert rescale(np.array([2.0, 3.0]), (0, 10), "data").tolist() == [0, 1]


def test_likelihoods_integrate_to_one():
    wide = np.linspace(-20, 20, 8001)                           # W itself stops at 4, inside the tails
    for mu, sigma in (((-1.0, 1.0), 1.0), ((-2.0, 0.0, 2.0), 1.5)):
        assert np.trapezoid(likelihoods(wide, mu, sigma), wide, axis=0) == pytest.approx([1] * len(mu), abs=1e-6)


# --- fig_likelihood_confidence --------------------------------------------------------------------------------------

def test_likelihood_confidence_ticks_and_trace_order():
    m = load_script("fig_likelihood_confidence")
    assert m.symmetric_ticks(W) == [-4, 0, 4]
    post2 = posterior(W)
    post3 = posterior(W, (-2.0, 0.0, 2.0), 1.5, (1 / 3,) * 3)
    assert [t[0] for t in m.confidence_traces(post2, "2AFC")] == ["Entropy model", "Difference model", "Max model"]
    assert [t[0] for t in m.confidence_traces(post3, "3AFC")] == ["Entropy model", "Difference model", "Max model",
                                                                 "Entropy Top2 model"]


# --- fig_decision_confidence_panels ---------------------------------------------------------------------------------

def test_decision_panels_two_option_bars_are_renormalised():
    m = load_script("fig_decision_confidence_panels")
    fig, ax = plt.subplots()
    m.plot_bars(ax, two_option=True, chosen_idx=0, highlight="dashed_box")
    heights = [p.get_height() for p in ax.patches if p.get_height() <= 1 and p.get_width() == m.width]
    assert heights == pytest.approx([0.625, 0.375])
    plt.close(fig)
    fig, ax = plt.subplots()
    m.plot_bars(ax, two_option=False, chosen_idx=1)
    assert [p.get_height() for p in ax.patches] == pytest.approx(m.values_all)
    plt.close(fig)


# --- fig_config_likelihoods -----------------------------------------------------------------------------------------

def test_config_likelihoods_constants_match_the_experiment():
    m = load_script("fig_config_likelihoods")
    assert m.SIGMA == SIG_S
    assert {cfg: list(mu) for cfg, mu in m.CONFIGS.items()} == {cfg: x.tolist() for cfg, x in CATEGORY_X.items()}


def test_config_likelihoods_dots_all_inside_the_axes():
    m = load_script("fig_config_likelihoods")
    rng = np.random.default_rng(m.SEED)
    mu = np.array(m.CONFIGS[m.DOTS_CFG])
    centres = np.column_stack([mu, np.full(len(mu), m.UY)])
    pts = (centres[:, None, :] + m.SIGMA * rng.standard_normal((len(mu), m.N_DOTS, 2))).reshape(-1, 2)
    assert len(pts) == 1125 and (np.abs(pts) <= m.LIM).all()


# --- fig_pooled_aic -------------------------------------------------------------------------------------------------

def test_pooled_aic_summary_reproduces_the_paper_numbers():
    tab = load_script("fig_pooled_aic").summary().set_index("model")
    assert (tab.n == 13).all()
    assert tab.k.tolist() == [6] * 6 + [7] * 4
    diff = tab.loc["diff"]
    assert diff.d_aic == 0 and diff.ci_lo == 0 and diff.ci_hi == 0 and diff.better == 0
    assert tab.d_aic["ent"] == pytest.approx(1937.0, abs=0.1)
    assert (tab.ci_lo["ent"], tab.ci_hi["ent"]) == pytest.approx((1353.4, 2550.3), abs=0.1)
    assert tab.d_aic["sym_top2_hard"] == pytest.approx(36.7, abs=0.1)
    assert (tab.ci_lo["sym_top2_hard"], tab.ci_hi["sym_top2_hard"]) == pytest.approx((-26.2, 89.5), abs=0.1)
    assert tab.d_aic["sym_top2_soft"] == pytest.approx(-25.7, abs=0.1)
    assert (tab.ci_lo["sym_top2_soft"], tab.ci_hi["sym_top2_soft"]) == pytest.approx((-79.8, 25.6), abs=0.1)
    assert tab.better["sym_top2_hard"] == 4 and tab.better["sym_top2_soft"] == 6
    assert tab.bar.tolist()[:3] == ["Diff", "Ent", "Sym Top-2"]
    assert tab.group.tolist() == ["No attention"] * 2 + ["Attention"] * 8
