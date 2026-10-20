import numpy as np

from bounded_optimal_confidence.models.readouts import diff_confidence, entropy_confidence, top2_entropy_confidence

RNG = np.random.default_rng(0)
POSTS = RNG.dirichlet(np.ones(3), size=1000)
FLOOR = 1.0 - 1.0 / np.log2(3)          # full-entropy confidence of a 50/50 pair


def test_ranges():
    for f in (diff_confidence, entropy_confidence, top2_entropy_confidence):
        c = f(POSTS)
        assert c.shape == (1000,) and np.all(c >= -1e-12) and np.all(c <= 1 + 1e-12)


def test_known_values():
    assert np.isclose(diff_confidence(np.array([0.5, 0.4, 0.1])), 0.1)
    assert np.isclose(entropy_confidence(np.full(3, 1 / 3)), 0.0)
    assert np.isclose(entropy_confidence(np.array([1.0, 0.0, 0.0])), 1.0)
    assert np.isclose(top2_entropy_confidence(np.array([0.5, 0.5, 0.0])), 0.0)
    assert np.isclose(entropy_confidence(np.array([0.5, 0.5, 0.0])), FLOOR)


def test_two_category_posteriors_full_entropy_is_top2_rescaled():
    """On a posterior with two non-zero categories (hard attention) the two entropy readouts are one scale apart."""
    two = POSTS.copy()
    two[:, 2] = 0
    two /= two.sum(axis=1, keepdims=True)
    full, top2 = entropy_confidence(two), top2_entropy_confidence(two)
    assert np.abs(full - (FLOOR + (1 - FLOOR) * top2)).max() < 1e-12
