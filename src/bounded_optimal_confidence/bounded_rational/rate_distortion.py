"""One-step rate distortion (single-variable Blahut-Arimoto) and the collapsed branch of the hierarchy it lifts to.

From ITBR_confidence/beta_sweep/objective_task.py. `onevar_ba` maximises E[U] - I(W;A)/beta3 over p(a|w) from a cold
uniform p(a); it is the rate-distortion frontier of the paper's figure. `collapsed_branch` is the same solution lifted
to a p(o|w) = p(o) fixed point of the three-variable map, recorded by the sweep as `J_collapsed`.
"""
import numpy as np

from bounded_optimal_confidence.bounded_rational import settings
from bounded_optimal_confidence.bounded_rational.blahut_arimoto import (expected_utility, marginalizeo, mutual_info_AO, mutual_info_WA_given_O,
                                                            mutual_info_XW, objective_value)
from bounded_optimal_confidence.bounded_rational.helpers import boltzmanndist_vectorized, effective_cardinality_from_pago


ONEVAR_MAXITER = 100000


ONEVAR_EPS_CONV = 1e-14


def onevar_ba(beta3, U_pre, pw):
    """Single-variable Blahut-Arimoto: maximise EU - I(W;A)/beta3 over p(a|w).

    This is the three-variable problem restricted to p(o|w) = p(o), where O is independent of W and
    the I(O;W) and I(A;O) terms both vanish. Returns (pa, pagw, n_iters).

    `boltzmanndist_vectorized` and the eps-guarded marginal below are the *same* calls
    compute_pagow_iteration and compute_marginals make at beta2 = inf, so this reaches the same
    fixed point the 3-var solver would reach on the invariant set rather than a lookalike of it.
    """
    card_a = U_pre.shape[0]
    pa = np.ones(card_a) / card_a
    n_iters = ONEVAR_MAXITER
    for i in range(ONEVAR_MAXITER):
        # p(a|w) ∝ p(a) exp(beta3 U(a, w)) -- compute_pagow_iteration with pago[:, k] = pa and
        # inv_beta2 = 0, which is what beta2 = inf gives.
        pagw = boltzmanndist_vectorized(pa[:, None], beta3, U_pre)
        pa_new = pagw @ pw
        # compute_marginals' eps guard, kept so the fixed point is bit-comparable with the
        # 3-var solver's rather than differing in the last digits by a floor it does not apply.
        pa_new = pa_new + np.finfo(float).eps
        pa_new /= np.sum(pa_new)
        diff = np.abs(pa_new - pa).max()
        pa = pa_new
        if diff < ONEVAR_EPS_CONV:
            n_iters = i + 1
            break
    pagw = boltzmanndist_vectorized(pa[:, None], beta3, U_pre)
    return pa, pagw, n_iters


def collapsed_branch(beta1, beta2, beta3, card_model, U_pre, pw):
    """The collapsed (p(o|w) = p(o)) solution, as the same 8-tuple run_ba returns.

    Built by solving the reduced 1-var problem and lifting: po and p(o|w) uniform, every p(a|o)
    column equal to p(a), every p(a|o,w) slice equal to the 1-var p(a|w). Verified to be a fixed
    point of the 3-var map to 1.3e-15 in state and 0.0 in J over 500 steps (V3).

    J is computed through objective_value and the same mutual_info_* functions the solver logs, not
    from the 1-var rate: the sweep's stored `objective` uses that accounting and the two must be
    comparable cell for cell. I(O;W) comes out 0 here, which is why J does not depend on beta1 --
    measured across the 743 collapsed cells of 20260815_112959, the spread within a beta3 column is
    <= 2.9e-9.
    """
    pa, pagw, n_iters = onevar_ba(beta3, U_pre, pw)
    card_w = U_pre.shape[1]

    po = np.ones(card_model) / card_model
    pogw = np.ones((card_model, card_w)) / card_model
    pago = np.repeat(pa[:, None], card_model, axis=1)
    pagow = np.repeat(pagw[:, None, :], card_model, axis=1)

    J = objective_value(
        expected_utility(U_pre, pogw, pagow, pw),
        mutual_info_XW(pogw, po, pw),
        mutual_info_AO(pago, pa, po),
        mutual_info_WA_given_O(pagow, pago, pogw, po, pw),
        beta1, beta2, beta3,
    )
    stop_info = {
        "stop_reason": "collapsed_branch",
        "n_iters": n_iters,
        "objective": float(J),
        "cycle_period": None,
        "cycle_J_drift": None,
        # An exact fixed point by construction; the 500-step drift measured in V3 is 1.3e-15.
        "diff_final": 0.0,
    }
    effective_cardinality = effective_cardinality_from_pago(pago, tol=settings.CARD_TOL)
    return (float(J), effective_cardinality, po, pa, pago, pogw,
            marginalizeo(pogw, pagow), stop_info)


def _collapsed_branch_applies(beta3):
    """Whether the collapsed branch is defined at this beta3.

    beta3 = 0 and beta3 = inf are the two degenerate ends the 3-var solver special-cases
    (compute_pagow_iteration branches on beta3 == 0), and the reduction's Boltzmann step is not the
    right object at either. Neither is in any swept range -- the grids run [1, 5] and [1, 10] -- so
    this only guards a hand call.
    """
    return np.isfinite(beta3) and beta3 > 0
