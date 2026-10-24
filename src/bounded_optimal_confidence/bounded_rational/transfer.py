"""Freeze a serial pathway (p(o|w), p(a|o)), re-derive p(a|o,w) on a configuration, score it through the joint.

From ITBR_confidence/generalisation/transfer.py (`objective` and `score`; the transfer experiment's aggregates and the
q optimisation are not carried over). `score` is the reference the sweep's own masked scoring path is checked against.
"""
import numpy as np

from bounded_optimal_confidence.bounded_rational import seeds as policies
from bounded_optimal_confidence.bounded_rational import world as gworld
from bounded_optimal_confidence.bounded_rational.info_terms import rate_utility, true_terms
from bounded_optimal_confidence.bounded_rational.reduced import marginalizeo_fast


def objective(terms, eu, beta1, beta2, beta3):
    """J = E[u] - I(W;O)/beta1 - I(A;O)/beta2 - I(W;A|O)/beta3, from TRUE information terms.

    Written out rather than routed through three_var_blahut_arimoto.objective_value because the
    terms are the true MIs, not the solver's, and passing them to the solver's assembler would
    invite the reader to assume they were the solver's. An infinite beta drops its term, matching
    objective_value's own convention.
    """
    def charge(mi, beta):
        return 0.0 if (beta == 0 or np.isinf(beta)) else float(mi) / float(beta)
    return float(eu - charge(terms["I(W;O)"], beta1) - charge(terms["I(A;O)"], beta2)
                 - charge(terms["I(W;A|O)"], beta3))


def score(pogw, pago, w_src, stimulus_config, betas, world_kw=None):
    """Score a frozen pathway on one target config.

    `w_src` is the grid p(o|w) was built on. If it differs from the target's, the router is
    interpolated across with edges held -- which happens for config 2 and nothing else, and the
    returned `regridded` flag says so per row rather than leaving it to be inferred.

    Returns E[U], J, the four true information terms, the effective number of actions the resulting
    p(a|w) actually uses, and the flag.
    """
    beta1, beta2, beta3 = betas
    world_d, U_pre, pw = gworld.build(stimulus_config, **(world_kw or {}))
    w_tgt = world_d["w_vals"]
    regridded = not gworld.same_grid(w_src, w_tgt)
    pogw_t = gworld.regrid_pogw(pogw, w_src, w_tgt) if regridded else np.asarray(pogw, dtype=float)

    state = policies.state_from(pogw_t, pago, beta2, beta3, U_pre, pw)
    pagow = state[4]
    pagw = marginalizeo_fast(pogw_t, pagow)
    terms = true_terms(pw, pogw_t, pagow)
    eu, i_wa = rate_utility(pagw, U_pre, pw)
    # true_terms' I(W;A) comes from the joint, rate_utility's from p(a|w) directly. They are the
    # same quantity by two routes; a gap means the marginalisation and the joint disagree, which
    # would make every other number here suspect.
    assert abs(i_wa - terms["I(W;A)"]) < 1e-9, \
        f"I(W;A) routes disagree by {i_wa - terms['I(W;A)']:.2e} on cfg {stimulus_config}"
    return {
        "eval_config": int(stimulus_config),
        "E_U": float(eu),
        "J": objective(terms, eu, beta1, beta2, beta3),
        "I_WO": float(terms["I(W;O)"]),
        "I_AO": float(terms["I(A;O)"]),
        "I_WAgO": float(terms["I(W;A|O)"]),
        "I_WA": float(terms["I(W;A)"]),
        "I_AOgW": float(terms["I(A;O|W)"]),
        "n_actions_used": int((pagw.max(axis=1) > 1e-3).sum()),
        "regridded": bool(regridded),
    }
