"""Numerical helpers of the solver: logs and Boltzmann distributions in nats, and the effective cardinality of p(a|o).

From ITBR_confidence/utils/helpers.py.
"""
import numpy as np


def log_nats(x):
    """Natural logarithm, handling zeros with epsilon.

    Nats, not bits: `boltzmanndist` exponentiates beta * utility with the natural exp, so every
    log that enters a Boltzmann utility -- and every information term scored against one -- has
    to be in the same units as that exponent, or beta silently means something different in the
    update than it does in the objective. See `three_var_blahut_arimoto.objective_value`.
    """
    return np.log(np.maximum(x, np.finfo(float).eps))


def boltzmanndist(p_prior, beta, utility):
    """Computes the Boltzmann distribution, handling infinite beta as argmax."""
    if np.isinf(beta):
        # Handle infinite beta: result is a hard max (1 at max utility, 0 elsewhere)
        # If multiple utilities are tied for max, they share probability equally
        max_u = np.max(utility)
        p_new = np.zeros_like(utility)
        is_max = (utility == max_u)
        p_new[is_max] = 1.0 / np.sum(is_max)
        return p_new

    # Standard stable softmax for finite beta
    exponent = beta * (utility - np.max(utility))
    p_new = p_prior * np.exp(exponent)
    return p_new / np.sum(p_new)


def boltzmanndist_vectorized(p_prior, beta, utility, axis=0):
    """Batched Boltzmann distribution, normalized along `axis`; p_prior and
    utility broadcast against each other on the remaining axes."""
    if np.isinf(beta):
        max_u = np.max(utility, axis=axis, keepdims=True)
        is_max = (utility == max_u)
        counts = np.sum(is_max, axis=axis, keepdims=True)
        return np.broadcast_to(np.where(is_max, 1.0 / counts, 0.0),
                                np.broadcast_shapes(p_prior.shape, utility.shape)).copy()

    exponent = beta * (utility - np.max(utility, axis=axis, keepdims=True))
    p_new = p_prior * np.exp(exponent)
    return p_new / np.sum(p_new, axis=axis, keepdims=True)


def effective_cardinality_from_pago(pago, tol=5e-3):
    """Number of behaviorally distinct model columns in p(a|o), by pairwise closeness."""
    card_o = pago.shape[1]
    parent = list(range(card_o))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(card_o):
        for j in range(i + 1, card_o):
            if np.abs(pago[:, i] - pago[:, j]).max() < tol:
                union(i, j)

    return len({find(k) for k in range(card_o)})
