"""Information terms of a policy or a hierarchy, in nats, computed from the joint p(a, o, w).

From ITBR_confidence/beta_sweep/info_terms.py. `true_terms` is the accounting every family in the sweep is scored by:
on a pinned policy the solver's own accounting reports the objective's charges rather than mutual informations, so the
figure's rate axes are built from the joint.
"""
import numpy as np

from bounded_optimal_confidence.bounded_rational.blahut_arimoto import mutual_info_AW


def _entropy(p):
    """Shannon entropy in nats of an array of probabilities, ignoring the zeros.

    Clamped at 0: `pw` sums to 1 + 4e-16 on this grid, so a single occupied bin lands at -6.7e-16
    rather than an exact zero. Entropy cannot be negative, so that roundoff is dropped -- but only
    the roundoff, since anything genuinely below zero is a bug.
    """
    p = np.asarray(p)
    p = p[p > 0]
    h = float(-(p * np.log(p)).sum())
    assert h > -1e-9, f"negative entropy {h}"
    return max(h, 0.0)


def rate_utility(pagw, U_pre, pw):
    """(E[u], I(W;A)) in nats for one policy -- its ACCURACY and its COMPLEXITY.

    `pagw` is (A, N), the shape the solver's p(a|w) has; `posteriors` returns (N, A), so the call
    site transposes. I(W;A) uses the policy's OWN marginal p(a) = sum_w p(w) p(a|w), through
    three_var_blahut_arimoto.mutual_info_AW -- the same function the solver logs.

    This is the twin of j_with_confidence_cost.fixed_terms, which computes the identical two
    numbers by the identical route. It is not imported from there because that module imports THIS
    one; instead `_check_tradeoffs` asserts the three fixed posteriors reproduce its published
    table, which catches a divergence rather than assuming there cannot be one.
    """
    pa = pagw @ pw
    pa = pa / pa.sum()
    eu = float(np.sum(pw * np.sum(pagw * U_pre, axis=0)))
    return eu, mutual_info_AW(pagw, pa, pw)


def true_terms(pw, pogw, pagow):
    """The information terms of a (possibly soft-routed) hierarchy, as TRUE MIs, in nats.

    Built from the joint p(a, o, w) = p(w) p(o|w) p(a|o,w), so every marginal is self-consistent.
    Cells with p(o|w) = 0 are masked before any log: their p(a|o,w) may be 0/0 (see
    `hierarchy_family`), and they carry zero weight.

    Returns I(W;O), I(A;O), I(W;A|O), I(W;A) and I(A;O|W). The last is the router's own noise
    reaching A: 0 for a hard cut, positive for a soft one, and the reason serial + residual nats do
    not add up to I(W;A) in general. The identity that ties them is asserted:
        I(W;A) = I(A;O) + I(W;A|O) - I(A;O|W)
    """
    live = pogw > 0
    pagow = np.where(live[None], pagow, 0.0)
    p_ow = pogw * pw[None, :]
    joint = pagow * p_ow[None]                                    # (A, O, N)
    po, p_ao = p_ow.sum(axis=1), joint.sum(axis=2)
    pa = p_ao.sum(axis=1)
    pagw = joint.sum(axis=1) / pw[None, :]
    h_a, h_o, h_ao = _entropy(pa), _entropy(po), _entropy(p_ao.ravel())
    h_o_w = float(-(p_ow * np.log(np.where(p_ow > 0, pogw, 1.0))).sum())
    h_a_ow = float(-(joint * np.log(np.where(joint > 0, pagow, 1.0))).sum())
    h_a_w = float(-(pw[None, :] * pagw * np.log(np.where(pagw > 0, pagw, 1.0))).sum())
    t = {"I(W;O)": h_o - h_o_w,
         "I(A;O)": h_a + h_o - h_ao,
         "I(W;A|O)": (h_ao - h_o) - h_a_ow,
         "I(W;A)": h_a - h_a_w,
         "I(A;O|W)": h_a_w - h_a_ow}
    gap = t["I(A;O)"] + t["I(W;A|O)"] - t["I(A;O|W)"] - t["I(W;A)"]
    assert abs(gap) < 1e-9, f"chain rule does not close: {gap:.2e}"
    return t
