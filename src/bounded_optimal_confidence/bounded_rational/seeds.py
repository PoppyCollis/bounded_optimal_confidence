"""Which p(o|w) routers the specialist solve starts from, how a router is completed into a full BA state, and
`state_from`, which completes a (p(o|w), p(a|o)) pair the same way.

From ITBR_confidence/fixed_points/init_policy.py (the seeds and completions; the random seeds are not carried over) and
ITBR_confidence/two_state_attention/policies.py (`state_from`).
"""
import numpy as np

from bounded_optimal_confidence.bounded_rational import settings
from bounded_optimal_confidence.bounded_rational.blahut_arimoto import compute_marginals, compute_pago_iteration, compute_pagow_iteration
from bounded_optimal_confidence.bounded_rational.reduced import hard_partition, solve_reduced


COMPLETIONS = ("solver", "reduced")


MIDSPLIT_FLOOR = 0.01


def decision_boundaries(mu_as):
    """The card-3 Bayes decision boundaries d_k = (mu_k + mu_{k+1}) / 2.

    Equal sigma across actions, so the boundaries are the midpoints of the means. These are the two
    places a psychometric slope is measured and the two cut positions the card-2 policy seeds.
    """
    mu_as = np.asarray(mu_as, dtype=float)
    return 0.5 * (mu_as[:-1] + mu_as[1:])


def exclusion_cut(mu_as):
    """(mu_1 + mu_A) / 2 -- the cut of the top-2 imitator, the router that drops the middle action.

    0.0 on any symmetric config. It always lies STRICTLY between d_1 and d_2, which is what makes
    the two Bayes boundaries a bracketing pair and is the whole argument for the card-2 seed set.
    """
    mu_as = np.asarray(mu_as, dtype=float)
    return 0.5 * (mu_as[0] + mu_as[-1])


def midsplit_pogw(U_pre, card_model):
    """The utility-argmax router with the outer categories SHARING a model state.

    w goes to o_1 where the MIDDLE category is the MAP action and to o_0 where an outer one is.
    NON-INTERVAL -- o_0's region is both tails at once -- so no set of cuts expresses it and an
    interval enumeration cannot reach it. It is in the set because regime_map measured exactly this
    structure winning by 4.5e-3 nats at (beta1 1000, beta2 inf, beta3 5), against a median
    adjacent-cell |dJ| of 2.3e-3.

    Defined off argmax_a U(a, w) rather than off hand-placed cuts, so it is the same object on any
    stimulus config and is exactly mirror-symmetric on a symmetric one.

    Returns None on a world where the middle category is never the MAP action -- then this is not a
    two-state router at all and the caller should skip it rather than solve a rank-1 problem.
    """
    card_a = U_pre.shape[0]
    map_action = U_pre.argmax(axis=0)
    inner = (map_action != 0) & (map_action != card_a - 1)
    pogw = np.zeros((card_model, U_pre.shape[1]))
    pogw[0, ~inner] = 1.0
    pogw[1, inner] = 1.0
    if pogw[0].sum() == 0 or pogw[1].sum() == 0:
        return None
    if card_model > 2:
        pogw = (pogw + MIDSPLIT_FLOOR) / np.sum(pogw + MIDSPLIT_FLOOR, axis=0)
    return pogw


def bayes_pogw(w_vals, mu_as, card_model):
    """One model state per stimulus category: hard cuts at every Bayes decision boundary.

    With equal sigma across actions this is exactly the argmax_a U(a, w) partition -- the same rule
    that generates `midsplit`, given enough states to keep the two tails apart instead of sharing
    one between them.

    Returns None unless there is exactly one model state per action, since with any other count the
    boundaries do not supply card_model - 1 cuts and "one state per category" is not a thing this
    router could be.
    """
    d = decision_boundaries(mu_as)
    if len(d) != card_model - 1:
        return None
    return hard_partition(w_vals, d, card_model)


def seed_pogws(card_model, U_pre, w_vals, mu_as):
    """label -> p(o|w) for every seed the policy prescribes on this world. Hard 0/1 routers.

    Hard rather than softened, deliberately: at finite beta1 the solver's own p(o|w) update softens
    them on the first iteration anyway, and starting hard means the seed is exactly the discrete
    structure it is named after rather than a temperature-dependent smear of it. The one exception
    is MIDSPLIT_FLOOR above.

    A label the policy names can still be absent from a given world -- `bayes` needs one model state
    per action, `midsplit` needs the middle category to be the MAP action somewhere -- so consumers
    iterate what comes back rather than assuming the set.
    """
    card_model = int(card_model)
    out = {}
    d = decision_boundaries(mu_as)
    if card_model == 2:
        # The bracketing pair. `d[0]` and `d[-1]` rather than `d[0]` and `d[1]` so a two-action
        # world (one boundary) contributes one cut instead of raising; there the exclusion cut IS
        # the single boundary and there is nothing to bracket.
        out["cut_d1"] = hard_partition(w_vals, float(d[0]), card_model)
        if len(d) > 1:
            out["cut_d2"] = hard_partition(w_vals, float(d[-1]), card_model)
    bayes = bayes_pogw(w_vals, mu_as, card_model)
    if bayes is not None:
        out["bayes"] = bayes
    mid = midsplit_pogw(U_pre, card_model)
    if mid is not None:
        out["midsplit"] = mid
    if not out:
        raise ValueError(f"no seed is defined for card_model {card_model} with "
                         f"{U_pre.shape[0]} actions -- the policy covers card 2 and "
                         f"card == n_actions")
    return out


def completion(beta3):
    """Which completion to use at this beta3. See the module docstring for the measurement."""
    return "solver" if float(beta3) == 0.0 else "reduced"


def complete_solver(pogw, beta2, beta3, U_pre, pw):
    """Complete a p(o|w) exactly the way threevarBAiterations completes its own init flags.

    Line-for-line with three_var_blahut_arimoto.py:530-541: uniform p(a|o,w), then
    compute_marginals, then ONE compute_pago_iteration, then the beta3 == 0 special case. Any
    deviation would put a difference between a seed and the solver's own cold start that is not the
    router, and comparisons against `sparse` rest on there being none.
    """
    num_acts = U_pre.shape[0]
    card_model = pogw.shape[0]
    pagow = np.ones((num_acts, card_model, U_pre.shape[1])) / num_acts
    po, pa, _ = compute_marginals(pw, pogw, pagow)
    pago = compute_pago_iteration(pogw, pagow, beta2, beta3, U_pre, pa, po, pw)
    if beta3 == 0:
        pagow = compute_pagow_iteration(pago, beta2, beta3, U_pre, pa)
    return (po, pa, np.array(pogw, dtype=float), pago, pagow)


def complete_reduced(pogw, beta2, beta3, U_pre, pw, alpha=None, eps_conv=None, maxiter=None,
                     patience=None):
    """Complete a p(o|w) by solving the reduced problem it defines, p(o|w) pinned.

    The four non-router components start at their exact optimum for that router, so no candidate is
    losing on its start rather than on its structure. beta1 never enters solve_reduced.

    The tolerances are arguments with config defaults rather than hard-wired, because
    two_state_attention/init_scheme.py passes its own and its stored runs must stay reproducible to
    the bit.
    """
    res = solve_reduced(
        pogw, beta2, beta3, U_pre, pw,
        alpha=settings.REDUCED_ALPHA if alpha is None else alpha,
        eps_conv=settings.REDUCED_EPS_CONV if eps_conv is None else eps_conv,
        maxiter=settings.REDUCED_MAXITER if maxiter is None else maxiter,
        patience=settings.REDUCED_DIFF_PATIENCE if patience is None else patience)
    pagow = res["pagow"]
    if beta3 == 0:
        # solve_reduced updates p(a|o,w) from the PREVIOUS p(a|o) (reduced_ba.py:201-202), so at
        # beta3 = 0 the pair it returns satisfies p(a|o,w) = p(a|o) only to eps_conv. The full
        # solver avoids this by reversing the order (three_var_blahut_arimoto.py:285). Rebuilt here
        # rather than in solve_reduced: partition_enum never runs at beta3 = 0, and changing the
        # shared solver would move results already on record.
        pagow = compute_pagow_iteration(res["pago"], beta2, beta3, U_pre, res["pa"])
    return (res["po"], res["pa"], np.array(pogw, dtype=float), res["pago"], pagow)


COMPLETERS = {"solver": complete_solver, "reduced": complete_reduced}


def init_states(card_model, beta2, beta3, U_pre, pw, w_vals, mu_as, both=False, **reduced_kw):
    """label -> init 5-tuple (po, pa, pogw, pago, pagow), ready for threevarBAiterations.

    Labels are `<seed>_<completion>`, so a row says which router it started from AND how the other
    four components were filled in, without the reader having to re-derive the completion from
    beta3.
    """
    seeds = seed_pogws(card_model, U_pre, w_vals, mu_as)
    which = COMPLETIONS if both else (completion(beta3),)
    out = {}
    for name in which:
        fn = COMPLETERS[name]
        kw = reduced_kw if name == "reduced" else {}
        for label, pogw in seeds.items():
            out[f"{label}_{name}"] = fn(pogw, beta2, beta3, U_pre, pw, **kw)
    return out


def state_from(pogw, pago, beta2, beta3, U_pre, pw):
    """Complete a (pogw, pago) pair into a full BA state.

    p(a|o,w) and the two marginals are *derived*, using the solver's own update functions, so the
    state is internally consistent in every component except the two that were specified. That
    matters for stability.py: any residual it then reports is attributable to pogw and pago, not
    to a hand-built p(a|o,w) that was never consistent with them in the first place.
    """
    card_a = U_pre.shape[0]
    pagow = compute_pagow_iteration(pago, beta2, beta3, U_pre, np.ones(card_a) / card_a)
    # pa is only used by compute_pagow_iteration through log(p(a|o)/p(a)), which vanishes at
    # beta2 = inf; at finite beta2 the placeholder above would bias it, so iterate the pair once
    # more against the pa the specified policy actually implies.
    po, pa, _ = compute_marginals(pw, pogw, pagow)
    pagow = compute_pagow_iteration(pago, beta2, beta3, U_pre, pa)
    po, pa, _ = compute_marginals(pw, pogw, pagow)
    return (po, pa, np.array(pogw, dtype=float), np.array(pago, dtype=float), pagow)
