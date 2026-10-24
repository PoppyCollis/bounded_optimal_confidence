"""The value of a state: the four information terms and J, through the solver's own metric functions.

From ITBR_confidence/fixed_points/score.py.
"""
from bounded_optimal_confidence.bounded_rational.blahut_arimoto import (expected_utility, mutual_info_AO, mutual_info_WA_given_O, mutual_info_XW,
                                                            objective_value)


def as_tuple(state):
    """(po, pa, pogw, pago, pagow) from either the state dict or the solver's own 5/6-tuple."""
    if isinstance(state, dict):
        return tuple(state[k] for k in ("po", "pa", "pogw", "pago", "pagow"))
    return tuple(state)[:5]


def info_terms(state, U_pre, pw):
    """(E[U], I(O;W), I(A;O), I(A;W|O)) for a state, in nats."""
    po, pa, pogw, pago, pagow = as_tuple(state)
    return (expected_utility(U_pre, pogw, pagow, pw),
            mutual_info_XW(pogw, po, pw),
            mutual_info_AO(pago, pa, po),
            mutual_info_WA_given_O(pagow, pago, pogw, po, pw))


def objective(terms, beta1, beta2, beta3):
    """J = E[U] - I(O;W)/b1 - I(A;O)/b2 - I(A;W|O)/b3, from stored terms."""
    return float(objective_value(*terms, beta1, beta2, beta3))


def score(state, U_pre, pw, beta1, beta2, beta3):
    """{'E_U', 'I_ow', 'I_ao', 'I_wago', 'J'} for a state."""
    terms = info_terms(state, U_pre, pw)
    return {"E_U": float(terms[0]), "I_ow": float(terms[1]), "I_ao": float(terms[2]),
            "I_wago": float(terms[3]), "J": objective(terms, beta1, beta2, beta3)}
