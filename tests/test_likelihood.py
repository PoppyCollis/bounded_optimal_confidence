import numpy as np
import pytest

from bounded_optimal_confidence.experiment import N_CATEGORIES, N_CONFIDENCE_LEVELS, load_trials
from bounded_optimal_confidence.likelihood.profile import decompose, exact_joint_scorer, profile_b_and_lapse
from bounded_optimal_confidence.likelihood.scoring import clean_nll, report_joint_probs
from bounded_optimal_confidence.models import simulate

TRIALS = load_trials(2, 1)
D_OBS, R_OBS = TRIALS.decision, TRIALS.confidence


def nll_from_joint(sim_d, sim_conf, edges, lapse):
    p = report_joint_probs(sim_d, sim_conf, np.sort(edges), lapse)
    return float(-np.log(p[np.arange(len(D_OBS)), D_OBS - 1, R_OBS - 1]).sum())


@pytest.mark.parametrize("model, log10_alpha", [("diff", 1.0), ("sym_spec_hard", 0.5), ("ent", np.inf)])
def test_exact_scorer_matches_report_joint_probs(model, log10_alpha):
    """Including degenerate triples, edges at 0 / 1 and point-mass confidences (alpha = inf on a hard model)."""
    sim_d, sim_conf = simulate(model, 1, TRIALS, 5.0, log10_alpha, None, 500, np.random.default_rng(0))
    score = exact_joint_scorer(sim_d, sim_conf, D_OBS - 1, R_OBS - 1, N_CATEGORIES, N_CONFIDENCE_LEVELS)
    rng = np.random.default_rng(1)
    triples = [np.sort(rng.random(3)) for _ in range(20)] + [np.array([0.0, 0.0, 1.0]), np.array([0.3, 0.3, 0.3]),
                                                             np.array([0.0, 0.5, 1.0])]
    for edges in triples:
        for lapse in (0.0, 0.02, 0.5):
            assert abs(score(edges, lapse) - nll_from_joint(sim_d, sim_conf, edges, lapse)) < 1e-9


def test_clean_nll_is_report_joint_probs_on_the_whole_draw():
    sim_d, sim_conf = simulate("ent", 1, TRIALS, 20.0, 1.25, None, 4500, np.random.default_rng(2))
    pos = [0]

    def chunk(n):
        out = sim_d[pos[0]:pos[0] + n], sim_conf[pos[0]:pos[0] + n]
        pos[0] += n
        return out

    edges, lapse = np.array([0.1, 0.4, 0.7]), 0.03
    assert abs(clean_nll(chunk, edges, lapse, D_OBS, R_OBS, 4500) - nll_from_joint(sim_d, sim_conf, edges, lapse)) < 1e-9


def test_profile():
    sim_d, sim_conf = simulate("sym_top2_soft", 1, TRIALS, 3.0, 1.0, 0.25, 1000, np.random.default_rng(3))
    res = profile_b_and_lapse(sim_d, sim_conf, D_OBS, R_OBS)
    assert np.all(np.diff(res.edges) >= 0) and np.all((res.edges >= 0) & (res.edges <= 1)) and 0 <= res.lapse <= 1
    assert abs(res.nll_joint - (res.nll_d + res.nll_rgd)) < 1e-9
    assert res.nll_d_min <= res.nll_d + 1e-12
    assert abs(res.nll_joint - nll_from_joint(sim_d, sim_conf, res.edges, res.lapse)) < 1e-9
    assert (res.nll_joint, res.nll_d, res.nll_rgd) == decompose(sim_d, sim_conf, res.edges, D_OBS, R_OBS,
                                                                N_CATEGORIES, N_CONFIDENCE_LEVELS, res.lapse)
    # a warm start at the optimum cannot make it worse
    again = profile_b_and_lapse(sim_d, sim_conf, D_OBS, R_OBS, x0=np.concatenate([res.edges, [res.lapse]]))
    assert again.nll_joint <= res.nll_joint + 1e-9
