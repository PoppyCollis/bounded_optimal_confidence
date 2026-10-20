import numpy as np
import pytest

from bounded_optimal_confidence.experiment import (CONFIGS, N_CATEGORIES, N_CONFIDENCE_LEVELS, SUBJECTS, load_subject,
                                                   load_trials)
from bounded_optimal_confidence.models import MODELS, get_model, simulate
from bounded_optimal_confidence.models.bayes import bayes_posterior
from bounded_optimal_confidence.models.simulate import dirichlet_argmax, model_posterior

TRIALS = load_trials(1, 3)


def test_registry():
    assert list(MODELS) == ["diff", "ent", "sym_top2_hard", "sym_spec_hard", "asym_spec_l_hard", "asym_spec_r_hard",
                            "sym_top2_soft", "sym_spec_soft", "asym_spec_l_soft", "asym_spec_r_soft"]
    assert [m.k for m in MODELS.values()] == [6] * 6 + [7] * 4
    assert get_model("diff") is MODELS["diff"] and get_model(MODELS["ent"]) is MODELS["ent"]
    with pytest.raises(ValueError):
        get_model("nope")


def test_data():
    for s in SUBJECTS:
        blocks = load_subject(s)
        assert [len(b) > 0 for b in blocks] == [True] * len(CONFIGS)
        for b in blocks:
            assert set(np.unique(b.decision)) <= set(range(1, N_CATEGORIES + 1))
            assert set(np.unique(b.confidence)) <= set(range(1, N_CONFIDENCE_LEVELS + 1))
            assert len(b.target_x) == len(b.target_y) == len(b.decision) == len(b.confidence)


@pytest.mark.parametrize("name", MODELS)
def test_simulate_shapes_and_determinism(name):
    lb = 0.5 if MODELS[name].fits_beta else None
    a = simulate(name, 3, TRIALS, 10.0, 1.0, lb, 2100, np.random.default_rng(1))
    b = simulate(name, 3, TRIALS, 10.0, 1.0, lb, 2100, np.random.default_rng(1))
    assert a[0].shape == a[1].shape == (2100, len(TRIALS))
    assert set(np.unique(a[0])) <= set(range(N_CATEGORIES))
    assert np.all((a[1] >= -1e-12) & (a[1] <= 1 + 1e-12))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


X = np.random.default_rng(2).normal(0, 120, size=5000)
POST = bayes_posterior(X, np.zeros_like(X), np.array([-96.0, -59.0, 96.0]), 0.0, 70.0)


@pytest.mark.parametrize("name", [n for n, m in MODELS.items() if m.attention == "hard"])
def test_hard_attention_keeps_at_most_two_categories(name):
    assert (model_posterior(name, POST) > 0).sum(axis=-1).max() <= 2


@pytest.mark.parametrize("name", [n for n, m in MODELS.items() if m.attention == "soft"])
def test_soft_attention_limits(name):
    hard = name.replace("_soft", "_hard")
    inf = model_posterior(name, POST, np.inf)
    if hard == "sym_top2_hard":     # the hard model is top-2 truncation: equal off the tie at x = 0
        assert np.abs(inf - model_posterior(hard, POST)).max() < 1e-12
    else:
        assert np.array_equal(inf, model_posterior(hard, POST))
    if name != "sym_top2_soft":     # partitions: beta = 1 is the Bayes posterior, i.e. Ent's
        assert np.abs(model_posterior(name, POST, 0.0) - POST).max() < 1e-12


def test_dirichlet_noise_keeps_zero_categories():
    post = model_posterior("asym_spec_l_hard", POST)
    noisy, d = dirichlet_argmax(post, 0.5, np.random.default_rng(3))
    assert np.all(noisy[post == 0] == 0)
    assert np.all(post[np.arange(len(d)), d] > 0)
    same, d_inf = dirichlet_argmax(post, np.inf, np.random.default_rng(3))
    assert same is post and np.array_equal(d_inf, post.argmax(axis=-1))
