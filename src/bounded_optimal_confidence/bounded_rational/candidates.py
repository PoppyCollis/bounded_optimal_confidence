"""The two hierarchies the sweep compares: the pinned top-2 pathway and the specialist solved on a configuration.

From ITBR_confidence/generalisation/candidates.py (`mirror_pathway` not carried over).
"""
import numpy as np

from bounded_optimal_confidence.bounded_rational import identity, settings
from bounded_optimal_confidence.bounded_rational import score as fp_score
from bounded_optimal_confidence.bounded_rational import seeds as init_policy
from bounded_optimal_confidence.bounded_rational import world as gworld
from bounded_optimal_confidence.bounded_rational.blahut_arimoto import threevarBAiterations
from bounded_optimal_confidence.bounded_rational.helpers import effective_cardinality_from_pago
from bounded_optimal_confidence.bounded_rational.rate_distortion import _collapsed_branch_applies, collapsed_branch
from bounded_optimal_confidence.bounded_rational.reduced import hard_partition


SEED_MODE = "policy"


def top2_pago(q=0.5, card_a=3):
    """p(a|o) for the top-2 code at mixing weight q: [[q, 0], [1-q, 1-q], [0, q]].

    Each model state keeps two actions and zeroes one outer extreme, putting q on the outer action
    it keeps and 1-q on the middle. q = 0.5 is `two_state_attention.policies.top2_pago`, exactly --
    test_agreement checks that, so this cannot drift from the object the rest of the repo tests.

    The zero is structural and survives every beta3: compute_pagow_iteration multiplies by this
    prior, so a zero entry stays exactly zero rather than becoming a small number.

    Refused for card_a != 3, because "the top 2 of 3" is what the object means.
    """
    if card_a != 3:
        raise ValueError(f"the top-2 code is defined for 3 actions, got {card_a}")
    q = float(q)
    if not 0.0 < q < 1.0:
        raise ValueError(f"q must lie strictly in (0, 1), got {q}")
    return np.array([[q, 0.0],
                     [1.0 - q, 1.0 - q],
                     [0.0, q]])


def top2_pathway(world_d, q=0.5):
    """The pinned top-2 serial pathway (pogw, pago) on this world.

    The cut is the exclusion cut (mu_1 + mu_A)/2, which is w = 0 on all four configs because every
    one has mu_1 = -mu_3. So this is rebuilt natively on each target grid rather than transported
    through `regrid_pogw` -- there is nothing config-specific in it to transport.

    That the hard cut IS the top-2 exclusion boundary, rather than an approximation to it, is
    argued in beta_sweep/visualise_top2_hierarchical.py: the category the top-2 rule discards is
    the one whose mean is farthest from w, the middle mean is never the farthest, so the discarded
    category switches exactly at (mu_1 + mu_3)/2 whatever mu_2 does.
    """
    pogw = hard_partition(world_d["w_vals"], world_d["exclusion_cut"],
                          card_model=world_d["card_model"])
    return pogw, top2_pago(q, card_a=3)


def _effective_cardinality(pogw, pago, tol=None):
    """min(routers used, behaviourally distinct p(a|o) columns) -- the sweeps' own definition."""
    tol = settings.CARD_TOL if tol is None else tol
    return int(min(int(np.any(np.asarray(pogw) != 0, axis=1).sum()),
                   effective_cardinality_from_pago(np.asarray(pago), tol=tol)))


def _solve_from(betas, init_state, world_d, U_pre, pw):
    """One full BA solve from a completed init state, reduced to what a row needs."""
    beta1, beta2, beta3 = betas
    po, pa, pogw, pago, pagow, pagw, _perf, stop = threevarBAiterations(
        world_d["card_model"], beta1, beta2, beta3, U_pre, pw,
        eps_conv=settings.EPS_CONV, maxiter=settings.FULL_MAXITER, init_state=init_state,
        verbose=False)
    st = identity.state_dict(po, pa, pogw, pago, pagow, pagw)
    out = {"state": st, "stop_reason": stop["stop_reason"], "n_iters": int(stop["n_iters"]),
           "diff_final": float(stop["diff_final"])}
    out.update(fp_score.score(st, U_pre, pw, beta1, beta2, beta3))
    out["J"] = float(stop["objective"])
    return out


def _collapsed(betas, world_d, U_pre, pw):
    """The collapsed (p(o|w) = p(o)) branch, constructed rather than iterated.

    A cell where this wins has no model state at all -- a "specialist" that is really one-step rate
    distortion. That is a real answer at that beta, not a failure, so it is recorded (the
    `branch_won` column) rather than filtered out. It also transfers trivially: a uniform router
    carries no information about w, so there is nothing config-specific in it to lose.
    """
    beta1, beta2, beta3 = betas
    if not _collapsed_branch_applies(beta3):
        return None
    J, _card, po, pa, pago, pogw, pagw, stop = collapsed_branch(
        beta1, beta2, beta3, world_d["card_model"], U_pre, pw)
    # collapsed_branch's return tuple drops pagow; rebuild it by the same repeat it used inside.
    pagow = np.repeat(np.asarray(pagw)[:, None, :], world_d["card_model"], axis=1)
    st = identity.state_dict(po, pa, pogw, pago, pagow, pagw)
    out = {"state": st, "stop_reason": stop["stop_reason"], "n_iters": int(stop["n_iters"]),
           "diff_final": float(stop["diff_final"])}
    out.update(fp_score.score(st, U_pre, pw, beta1, beta2, beta3))
    out["J"] = float(J)
    return out


def solve_specialist(stimulus_config, betas, world_kw=None, include_collapsed=True):
    """The best hierarchy this config supports at these betas, as a frozen pathway plus its record.

    Returns a dict with `pogw`, `pago` (the pathway to freeze), `w_vals` (the grid they live on,
    needed to regrid them onto another config), `J` (the value it achieves at home, which the
    self-transfer check must reproduce), and the provenance a row needs: which seed won, the stop
    reason, the effective cardinality, whether the collapsed branch beat every split seed, and the
    chirality.

    Seeds + collapsed, best J. Ties are broken by `max`'s first-wins, which on a mirror-symmetric
    config means the winning lean is decided by seed order rather than by the problem -- that is
    what `chirality` is recorded for, and what `mirror_pathway` exists to complete.

    `include_collapsed=False` drops the card-1 branch from the competition, so what comes back is
    the best SPLIT (card 2) fixed point whether or not the collapsed one would have scored higher.
    That is the right object when the question fixes the cardinality -- a curve in beta3 that
    silently swapped families mid-axis would not be a curve of one thing. The default keeps the
    branch, because every other caller wants the best solution at these betas, full stop.
    """
    world_d, U_pre, pw = gworld.build(stimulus_config, **(world_kw or {}))
    beta1, beta2, beta3 = betas
    inits = init_policy.init_states(world_d["card_model"], beta2, beta3, U_pre, pw,
                                    world_d["w_vals"], world_d["mu_as"])
    results = {label: _solve_from(betas, st, world_d, U_pre, pw) for label, st in inits.items()}
    col = _collapsed(betas, world_d, U_pre, pw) if include_collapsed else None
    if col is not None:
        results["collapsed"] = col

    label = max(results, key=lambda k: results[k]["J"])
    best = results[label]
    st = best["state"]
    card = _effective_cardinality(st["pogw"], st["pago"])
    chir, sk, cent = identity.chirality(st, world_d["w_vals"], world_d["mirror_symmetric"], card)
    spread = float(max(r["J"] for r in results.values()) - min(r["J"] for r in results.values()))
    return {
        "stimulus_config": int(stimulus_config),
        "pogw": np.array(st["pogw"], dtype=float),
        "pago": np.array(st["pago"], dtype=float),
        "w_vals": np.array(world_d["w_vals"], dtype=float),
        "J": float(best["J"]),
        "E_U": float(best["E_U"]),
        "seed": label,
        "branch_won": "collapsed" if label == "collapsed" else "split",
        "stop_reason": best["stop_reason"],
        "n_iters": int(best["n_iters"]),
        "diff_final": float(best["diff_final"]),
        "effective_cardinality": card,
        "chirality": int(chir),
        "skew": float(sk),
        "crossing_centroid": float(cent),
        "J_spread_seeds": spread,
        "n_seeds": len(results),
        "mirror_symmetric": bool(world_d["mirror_symmetric"]),
    }
