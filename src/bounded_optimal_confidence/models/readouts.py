"""Confidence readouts: a posterior (..., K) -> a continuous confidence in [0, 1].

The confidence boundaries b1 <= b2 <= b3 (fitted) then bin it into the four reported levels.
"""
import numpy as np

from bounded_optimal_confidence.models.attention import top2_truncate

EPS = np.finfo(float).eps


def diff_confidence(post):
    """Diff: the top posterior probability minus the second."""
    srt = np.sort(post, axis=-1)
    return srt[..., -1] - srt[..., -2]


def entropy_confidence(post):
    """Entropy: 1 - H(p) / log2 K. 0 at the uniform posterior, 1 at a point mass."""
    p = np.where(post == 0, EPS, post)
    h = -np.sum(p * np.log2(p), axis=-1)
    return 1.0 - h / np.log2(post.shape[-1])


def top2_entropy_confidence(post):
    """Entropy over the two most probable categories: 1 - H(top-2 renormalised) in bits (log2 2 = 1).

    The hard-attention models' posterior has at most two non-zero categories, so this is their entropy readout on
    the two-category scale [0, 1]; full entropy would be the same number rescaled onto [1 - 1/log2 3, 1].
    """
    p = top2_truncate(post)
    p = np.where(p == 0, EPS, p)
    h = -np.sum(p * np.log2(p), axis=-1)
    return 1.0 - h


READOUTS = {"diff": diff_confidence, "entropy": entropy_confidence, "top2_entropy": top2_entropy_confidence}
