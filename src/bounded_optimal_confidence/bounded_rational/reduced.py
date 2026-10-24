"""Blahut-Arimoto with p(o|w) held fixed: the reduced solve that completes a seed router, and the einsum p(a|w).

From ITBR_confidence/partition_enum/reduced_ba.py. Pinning p(o|w) removes the beta1 update, so the remaining four
components (p(a|o,w), p(a|o), p(a), p(o)) form a smooth, beta1-free fixed-point problem; at a full fixed point the
reduced solution IS the full one. The tolerances (REDUCED_*) are in `settings`.
"""
import numpy as np

from bounded_optimal_confidence.bounded_rational import settings
from bounded_optimal_confidence.bounded_rational.blahut_arimoto import (compute_marginals, compute_pago_iteration, compute_pagow_iteration, expected_utility,
                                                            marginalizeo, mutual_info_AO, mutual_info_WA_given_O, mutual_info_XW,
                                                            objective_value)


def marginalizeo_fast(pogw, pagow):
    """p(a|w) = sum_o p(o|w) p(a|o,w). einsum form of marginalizeo."""
    return np.einsum("akj,kj->aj", pagow, pogw)


def _marginals_fast(pw, pogw, pagow):
    """compute_marginals with the einsum p(a|w), keeping its eps guards and renormalisation."""
    po = pogw @ pw
    po = po + np.finfo(float).eps
    po = po / np.sum(po)

    pagw = marginalizeo_fast(pogw, pagow)

    pa = pagw @ pw
    pa = pa + np.finfo(float).eps
    pa = pa / np.sum(pa)
    return po, pa, pagw


def hard_partition(w_vals, boundary, card_model=2):
    """p(o|w) as a hard partition of w into `card_model` contiguous intervals.

    `boundary` is a scalar (card_model == 2) or card_model - 1 cut points in w units. Cuts are
    right-closed -- np.searchsorted's default side="left" puts a w exactly equal to a cut in the
    region *below* it -- so region k is (cut[k-1], cut[k]]. Which side a single grid point falls
    on is immaterial to anything here; it is written down only so the enumeration's cut index and
    the region it names cannot drift apart.
    """
    w_vals = np.asarray(w_vals, dtype=float)
    cuts = np.atleast_1d(np.asarray(boundary, dtype=float))
    if len(cuts) != card_model - 1:
        raise ValueError(f"card_model {card_model} needs {card_model - 1} cuts, got {len(cuts)}")
    idx = np.searchsorted(np.sort(cuts), w_vals)
    pogw = np.zeros((card_model, len(w_vals)))
    pogw[idx, np.arange(len(w_vals))] = 1.0
    return pogw


def _objective(po, pa, pogw, pago, pagow, U_pre, pw, beta2, beta3):
    """J for the reduced state, via the same helpers _state_metrics uses.

    beta1 is passed as inf because I(O;W) is a property of the pinned p(o|w) and is therefore a
    constant of the enumeration, not something the reduced problem trades against. J is reported
    for provenance only -- the enumeration minimises a residual, never J -- so any other beta1
    would shift every row by the same constant.
    """
    I_ow = mutual_info_XW(pogw, po, pw)
    I_ao = mutual_info_AO(pago, pa, po)
    I_wago = mutual_info_WA_given_O(pagow, pago, pogw, po, pw)
    EU = expected_utility(U_pre, pogw, pagow, pw)
    return float(objective_value(EU, I_ow, I_ao, I_wago, np.inf, beta2, beta3))


def solve_reduced(pogw, beta2, beta3, U_pre, pw, alpha=None, eps_conv=None, maxiter=None,
                  patience=None, init=None, with_objective=False):
    """Iterate (pagow, pago, pa) to a fixed point with p(o|w) pinned to `pogw`.

    Returns a dict with po, pa, pago, pagow, pagw, n_iters, stop_reason, diff and (optionally)
    objective.

    `init` may be a previous result, used to warm-start pagow/pago/pa. The enumeration walks the
    boundary axis in order, so consecutive problems are near-identical and the warm start cuts
    iterations substantially. Unlike in the full solver, warm starting cannot change *which*
    fixed point is reached: the branch structure lives in the pinned p(o|w) update.
    """
    alpha = settings.REDUCED_ALPHA if alpha is None else alpha
    eps_conv = settings.REDUCED_EPS_CONV if eps_conv is None else eps_conv
    maxiter = settings.REDUCED_MAXITER if maxiter is None else maxiter
    patience = settings.REDUCED_DIFF_PATIENCE if patience is None else patience

    card_a, card_w = U_pre.shape

    # p(o) is a function of the pinned p(o|w) alone, so it is constant across the iteration and
    # is computed once instead of inside _marginals_fast every step.
    po = pogw @ pw
    po = po + np.finfo(float).eps
    po = po / np.sum(po)

    if init is not None:
        pagow = np.array(init["pagow"], dtype=float)
        pago = np.array(init["pago"], dtype=float)
        pa = np.array(init["pa"], dtype=float)
    else:
        # Same initialisation the full solver uses (threevarBAiterations: uniform pagow, then one
        # pago update off it), minus the pogw init, which is the thing being pinned.
        pagow = np.ones((card_a, pogw.shape[0], card_w)) / card_a
        _, pa, _ = _marginals_fast(pw, pogw, pagow)
        pago = compute_pago_iteration(pogw, pagow, beta2, beta3, U_pre, pa, po, pw)

    stop_reason, n_iters, diff, streak = "maxiter", maxiter, np.inf, 0
    for i in range(maxiter):
        pagow_new = compute_pagow_iteration(pago, beta2, beta3, U_pre, pa)
        pago_new = compute_pago_iteration(pogw, pagow_new, beta2, beta3, U_pre, pa, po, pw)
        _, pa_new, _ = _marginals_fast(pw, pogw, pagow_new)

        # Raw residual, before under-relaxation, so it stays alpha-free -- same convention as
        # _ba_step (three_var_blahut_arimoto.py:251).
        diff = max(np.abs(pagow_new - pagow).max(),
                   np.abs(pago_new - pago).max(),
                   np.abs(pa_new - pa).max())

        if alpha < 1.0:
            pa_new = (1 - alpha) * pa + alpha * pa_new
            pago_new = (1 - alpha) * pago + alpha * pago_new

        pagow, pago, pa = pagow_new, pago_new, pa_new

        streak = streak + 1 if diff < eps_conv else 0
        if streak >= patience:
            stop_reason, n_iters = "diff", i + 1
            break

    pagw = marginalizeo_fast(pogw, pagow)
    out = {"po": po, "pa": pa, "pago": pago, "pagow": pagow, "pagw": pagw,
           "n_iters": n_iters, "stop_reason": stop_reason, "diff": float(diff)}
    if with_objective:
        out["objective"] = _objective(po, pa, pogw, pago, pagow, U_pre, pw, beta2, beta3)
    return out
