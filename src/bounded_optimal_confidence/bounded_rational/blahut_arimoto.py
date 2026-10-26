"""The three-variable Blahut-Arimoto solver of the bounded-rational hierarchy p(o|w), p(a|o), p(a|o,w).

From ITBR_confidence/three_var_blahut_arimoto.py: the update functions, the information terms (nats), the objective,
one BA step, the limit-cycle detector and `threevarBAiterations`. The solver constants (ALPHA, the patience counters,
the cycle detector's thresholds) live in `settings`; the comments on why they take their values are there too.

This code is based on the three-variable Blahut-Arimoto optimisation of Tim Genewein's
https://github.com/tgenewein/BoundedRationalityAbstractionAndHierarchicalDecisionMaking (Genewein, Leibfried,
Grau-Moya and Braun, 2015, "Bounded rationality, abstraction, and hierarchical decision-making: an
information-theoretic optimality principle", Frontiers in Robotics and AI 2:27), ported from Julia to numpy.
"""
import hashlib
import time
from collections import deque

import numpy as np
import pandas as pd

from bounded_optimal_confidence.bounded_rational import settings
from bounded_optimal_confidence.bounded_rational.helpers import boltzmanndist, boltzmanndist_vectorized, log_nats


EPS = np.finfo(float).eps


def marginalizeo(pogw, pagow):
    card_a, card_o, card_w = pagow.shape
    pagw = np.zeros((card_a, card_w))
    for j in range(card_w):
        # p(a|w) = ∑_o p(o|w)p(a|o,w)
        pagw[:, j] = pagow[:, :, j] @ pogw[:, j]
    return pagw


def compute_marginals(pw, pogw, pagow):
    # p(o) = ∑_w p(o|w)p(w)
    po = pogw @ pw
    po += np.finfo(float).eps
    po /= np.sum(po)

    # p(a|w)
    pagw = marginalizeo(pogw, pagow)

    # p(a) = ∑_w p(a|w)p(w)
    pa = pagw @ pw
    pa += np.finfo(float).eps
    pa /= np.sum(pa)

    return po, pa, pagw


def compute_pago_iteration(pogw, pagow, beta2, beta3, U_pre, pa, po, pw):
    card_a, card_w = U_pre.shape
    card_o = pogw.shape[0]
    pago = np.zeros((card_a, card_o))

    for k in range(card_o):
        # p(w|o=k) = p(o|w)p(w)/p(o)
        pwgo_k = (pogw[k, :] * pw) / po[k]

        if beta3 == 0:
            # Sequential case: p(a|o) ∝ p(a)exp( β2 ∑_w p(w|o)U(a,w) )
            pago[:, k] = boltzmanndist(pa, beta2, U_pre @ pwgo_k)
        else:
            # General case: p(a|o) = ∑_w p(w|o)p(a|o,w)
            # Use dot product for pagow[:, k, :] * pwgo_k
            pago[:, k] = pagow[:, k, :] @ pwgo_k
            pago[:, k] += np.finfo(float).eps
            pago[:, k] /= np.sum(pago[:, k])

    return pago


def compute_pagow_iteration(pago, beta2, beta3, U_pre, pa):
    # vectorized version of compute_pagow_iteration_old
    card_a, card_w = U_pre.shape
    card_o = pago.shape[1]

    inv_beta2 = 0.0 if np.isinf(beta2) else (1.0 / beta2)

    if beta3 == 0:
        return np.broadcast_to(pago[:, :, None], (card_a, card_o, card_w)).copy()

    # log_nats(pago[:, k] / pa) doesn't depend on j -- compute once per k, not per (k, j)
    log_term = log_nats(pago / pa[:, None])  # (card_a, card_o)
    pagow_util = U_pre[:, None, :] - inv_beta2 * log_term[:, :, None]  # (card_a, card_o, card_w)
    return boltzmanndist_vectorized(pago[:, :, None], beta3, pagow_util)


def compute_pogw_iteration(pago, pagow, beta1, beta2, beta3, U_pre, pa, po):
    # vectorized version of compute_pogw_iteration_old
    card_o, card_w = pagow.shape[1], pagow.shape[2]

    inv_beta2 = 0.0 if (beta2 == 0 or np.isinf(beta2)) else (1.0 / beta2)
    inv_beta3 = 0.0 if (beta3 == 0 or np.isinf(beta3)) else (1.0 / beta3)

    pagow_c = np.clip(pagow, EPS, 1.0)
    pago_c = np.clip(pago, EPS, 1.0)
    pa_c = np.clip(pa, EPS, 1.0)

    # D_KL( p(a|o,w) || p(a|o) ), same expression as `inner` in mutual_info_WA_given_O
    DKL_ago = np.sum(pagow_c * (np.log(pagow_c) - np.log(pago_c[:, :, None])), axis=0)  # (card_o, card_w)

    if beta3 == 0:
        if np.sum(DKL_ago) > 1e-8:
            raise ValueError(f"Sequential case violation: D_KL is {np.sum(DKL_ago)}")
        EU_cond = pago.T @ U_pre  # (card_o, card_w)
        DKL_a = np.sum(pago_c * (np.log(pago_c) - np.log(pa_c)[:, None]), axis=0)  # (card_o,)
        pogw_util = EU_cond - inv_beta2 * DKL_a[:, None]
    else:
        EU_cond = np.sum(pagow * U_pre[:, None, :], axis=0)  # (card_o, card_w)
        DKL_a = np.sum(pagow_c * (np.log(pagow_c) - np.log(pa_c)[:, None, None]), axis=0)  # (card_o, card_w)
        pogw_util = EU_cond - inv_beta2 * DKL_a - (inv_beta3 - inv_beta2) * DKL_ago

    return boltzmanndist_vectorized(po[:, None], beta1, pogw_util)


def mutual_info_XW(pogw, po, pw):
    # I(O;W) -- vectorized version of mutual_info_XW_old
    p = np.clip(pogw, EPS, 1.0)
    logpo = np.log(np.clip(po, EPS, 1.0))[:, None]
    return float(np.sum(pw[None, :] * np.sum(p * (np.log(p) - logpo), axis=0)))


def mutual_info_AO(pago, pa, po):
    # I(A;O)
    A, X = pago.shape
    total = 0.0
    for x in range(X):
        pax = np.clip(pago[:, x], EPS, 1.0)
        total += po[x] * np.sum(pax * (np.log(pax) - np.log(np.clip(pa, EPS, 1.0))))
    return float(total)


def mutual_info_WA_given_O(pagow, pago, pogw, po, pw):
    # I(A;W|O) -- vectorized version of mutual_info_WA_given_O_old
    p_axw = np.clip(pagow, EPS, 1.0)
    pax = np.clip(pago, EPS, 1.0)[:, :, None]
    # Bayes: p(w|o) = p(o|w)p(w)/p(o)
    w_post = (pogw * pw[None, :]) / (po[:, None] + EPS)
    inner = np.sum(p_axw * (np.log(p_axw) - np.log(pax)), axis=0)
    return float(np.sum(po[:, None] * w_post * inner))


def mutual_info_AW(pagw, pa, pw):
    # I(A;W) -- vectorized version of mutual_info_AW_old
    p = np.clip(pagw, EPS, 1.0)
    logpa = np.log(np.clip(pa, EPS, 1.0))[:, None]
    return float(np.sum(pw[None, :] * np.sum(p * (np.log(p) - logpa), axis=0)))


def expected_utility(U_pre, pogw, pagow, pw):
    # vectorized version of expected_utility_old
    contrib = np.sum(pagow * U_pre[:, None, :], axis=0)
    return float(np.sum(pw[None, :] * pogw * contrib))


def objective_value(EU, I_ow, I_ao, I_wago, beta1, beta2, beta3):
    """J = E[U] - I(O;W)/beta1 - I(A;O)/beta2 - I(A;W|O)/beta3.

    UNITS: everything here is in nats, and it has to be. U_pre is a Gaussian log-likelihood
    built with the natural log, and `boltzmanndist` raises `np.exp` to beta * utility -- so the
    quantity the iteration actually trades utility against is the natural-log information, and
    beta is a nats-per-nat exchange rate. Scoring the same betas against base-2 information
    (as this did until 2026-09-02) understates every information term by ln 2 ~ 0.693 relative
    to the exponent that produced it, and the same mismatch was in the update utilities: the
    log(p(a|o)/p(a)) correction in compute_pagow_iteration and the two KLs in
    compute_pogw_iteration were base 2 inside a natural exp. Both are natural log now, and
    `mutual_info_*` return nats to match. Consequence: J and the reported information terms
    change, and so does the meaning of a fitted beta -- a beta fit under the old code
    corresponds to roughly beta / ln 2 here. Any beta fit or sweep run before this needs
    re-running, not rescaling; the mismatch was not a single consistent rescaling of the
    betas, so the old iteration was not solving this objective at any beta.
    """
    #inv_b1 = 0.0 if np.isinf(beta1) else 1.0 / beta1
    inv_b1 = 0.0 if (beta1 == 0 or np.isinf(beta1)) else (1.0 / beta1)
    
    # inv_b2 = 0.0 if np.isinf(beta2) else 1.0 / beta2
    inv_b2 = 0.0 if (beta2 == 0 or np.isinf(beta2)) else (1.0 / beta2)

    #inv_b3 = 0.0 if (beta3 == 0 or np.isinf(beta3)) else 1.0 / beta3
    inv_b3 = 0.0 if (beta3 == 0 or np.isinf(beta3)) else (1.0 / beta3)
    
    return EU - (inv_b1 * I_ow) - (inv_b2 * I_ao) - (inv_b3 * I_wago)


def _ba_step(state, beta1, beta2, beta3, U_pre, pw, alpha):
    """One Blahut-Arimoto update of (po, pa, pogw, pago, pagow).

    Returns (new_state, pagw, diff), where `diff` is the largest absolute change across all
    the marginals, measured against the *raw* (undamped) updates for po/pa/pago so that the
    residual stays alpha-free -- see settings.EPS_CONV.

    Factored out of the main loop because the best-phase replay has to reproduce the same
    trajectory exactly; sharing this function is what makes that true by construction rather
    than by two copies staying in sync. Every array returned is freshly allocated (none of the
    compute_* helpers write in place), so callers may hold a reference to a state without
    copying it.
    """
    po, pa, pogw, pago, pagow = state

    pogw_new = compute_pogw_iteration(pago, pagow, beta1, beta2, beta3, U_pre, pa, po)
    if beta3 == 0:
        pago_new = compute_pago_iteration(pogw, pagow, beta2, beta3, U_pre, pa, po, pw)
    else:
        pagow_new = compute_pagow_iteration(pago, beta2, beta3, U_pre, pa)
        pago_new = compute_pago_iteration(pogw, pagow_new, beta2, beta3, U_pre, pa, po, pw)

    # the exact p(a|o) update, before any under-relaxation is applied
    pago_raw = pago_new

    # under-relaxation of p(a|o). Applied here rather than below the marginals because at
    # beta3 = 0 the p(a|o,w) built from it has to be the one that ends up in the state: the
    # sequential case carries the structural constraint p(a|o,w) = p(a|o), and
    # compute_pogw_iteration rejects a state that violates it on the next step. po and pa do
    # not enter either update, so relaxing p(a|o) before them changes nothing for beta3 != 0.
    if alpha < 1.0:
        pago_new = (1 - alpha) * pago + alpha * pago_new

    if beta3 == 0:
        # p(a|o,w) = p(a|o), built from the relaxed p(a|o) that the state will actually hold
        pagow_new = compute_pagow_iteration(pago_new, beta2, beta3, U_pre, pa)

    po_new, pa_new, pagw = compute_marginals(pw, pogw_new, pagow_new)

    # the exact marginal updates, before any under-relaxation is applied
    po_raw, pa_raw = po_new, pa_new

    # under-relaxation of the remaining coupling marginals (po, pa)
    if alpha < 1.0:
        po_new = (1 - alpha) * po + alpha * po_new
        pa_new = (1 - alpha) * pa + alpha * pa_new

    diff = max(
        np.abs(pogw_new - pogw).max(),   # unrelaxed: applied == raw
        # unrelaxed, except at beta3 = 0 where p(a|o,w) tracks the relaxed p(a|o). That term
        # is then alpha times the pago_raw residual below, so it never sets the max.
        np.abs(pagow_new - pagow).max(),
        np.abs(po_raw - po).max(),        # raw residual, alpha-free
        np.abs(pa_raw - pa).max(),
        np.abs(pago_raw - pago).max(),
    )
    return (po_new, pa_new, pogw_new, pago_new, pagow_new), pagw, diff


def _state_metrics(state, pagw, U_pre, pw, beta1, beta2, beta3):
    """The scalar row logged per iteration, for the state as it is *after* that iteration's
    update. Everything that needs J -- the history, the windowed-J criterion and the cycle
    detector -- reads this one computation, so they cannot disagree and none of it is done
    twice."""
    po, pa, pogw, pago, pagow = state
    I_ow = mutual_info_XW(pogw, po, pw)
    I_ao = mutual_info_AO(pago, pa, po)
    I_wago = mutual_info_WA_given_O(pagow, pago, pogw, po, pw)
    I_aw = mutual_info_AW(pagw, pa, pw)
    EU = expected_utility(U_pre, pogw, pagow, pw)
    return dict(
        I_ow=I_ow, I_ao=I_ao, I_wago=I_wago, I_aw=I_aw, E_U=EU,
        Objective_value=objective_value(EU, I_ow, I_ao, I_wago, beta1, beta2, beta3),
    )


def _assignment_fingerprint(pogw):
    """Digest of the hard p(o|w) support.

    At beta1 = inf every entry of p(o|w) is exactly 0 or 1/(number of tied maxima), so the
    support *is* the discrete state of the iteration and its exact recurrence is what a cycle
    consists of. Hashing ~3 KB per iteration is nothing against a ~4.7 ms iteration.
    """
    return hashlib.blake2b((pogw > 0).astype(np.uint8).tobytes(), digest_size=16).digest()


class _Ring:
    """Fixed-capacity ring buffer with O(1) lookback. `lag(k)` is the value from k iterations
    before the most recent one."""

    __slots__ = ("_buf", "_n")

    def __init__(self, capacity):
        self._buf = [None] * capacity
        self._n = 0

    def append(self, value):
        self._buf[self._n % len(self._buf)] = value
        self._n += 1

    def has_lag(self, k):
        return 0 <= k < min(self._n, len(self._buf))

    def lag(self, k):
        return self._buf[(self._n - 1 - k) % len(self._buf)]


class _CycleDetector:
    """Detects an exact limit cycle in the beta1 = inf assignment sequence, in O(1) per
    iteration, and reports when that cycle has become stationary.

    Two separate questions, and both have to be answered before a cycle may be used as a
    stopping rule:

    1. *What is the period?* Confirmed by `f[t] == f[t - P]` holding for CYCLE_MIN_PERIODS
       consecutive full periods -- NOT by "this fingerprint last occurred P iterations ago".
       That distinction is load-bearing: assignments repeat *within* a period (22 distinct
       assignments over a period of 99 on one measured cell), so a last-seen-lag test keeps
       proposing sub-periods. Here `last_seen` only ever *proposes* a candidate; a wrong
       proposal fails the very next equality check and a fresh one is taken, so the detector
       self-corrects onto the true fundamental period.

    2. *Has it settled?* A period can lock in while the continuous part of the state is still
       drifting (measured on three cells: period confirmed, per-phase J drift still 1e-4 to
       5e-1). So the cycle also has to be stationary: |J[t] - J[t-P]| < CYCLE_J_EPS for a full
       period. Settled orbits sit at exactly 0.0, unsettled ones at >= 3e-5.

    The stationarity test is also what makes this safe at finite beta1, where the support is
    constant and P = 1 is proposed immediately: periods below 2 are rejected outright (a fixed
    point is the diff criterion's business), and no longer period can pass without J genuinely
    repeating to 1e-9.
    """

    def __init__(self, max_period=None, min_periods=None, j_eps=None):
        self.max_period = settings.CYCLE_MAX_PERIOD if max_period is None else max_period
        self.min_periods = settings.CYCLE_MIN_PERIODS if min_periods is None else min_periods
        self.j_eps = settings.CYCLE_J_EPS if j_eps is None else j_eps
        self._fps = _Ring(self.max_period + 1)
        self._Js = _Ring(self.max_period + 1)
        self._last_seen = {}
        self._t = -1
        self._candidate = None      # period currently under test
        self._streak = 0            # consecutive iterations with f[t] == f[t - candidate]
        self._stationary = 0        # consecutive iterations with j_drift < j_eps
        self.period = None          # confirmed period, or None
        self.j_drift = None         # |J[t] - J[t - period]| at the latest iteration

    def update(self, fingerprint, J):
        """Feed one post-update iterate. Returns True once a stationary cycle of period >= 2
        is confirmed, i.e. once the caller should stop and pick the best phase."""
        self._t += 1
        self._fps.append(fingerprint)
        self._Js.append(J)

        # verify the standing candidate, or take a new proposal from the last recurrence
        if (self._candidate is not None and self._fps.has_lag(self._candidate)
                and self._fps.lag(self._candidate) == fingerprint):
            self._streak += 1
        else:
            previous = self._last_seen.get(fingerprint)
            gap = None if previous is None else self._t - previous
            self._candidate = gap if gap is not None and gap <= self.max_period else None
            self._streak = 0 if self._candidate is None else 1

        self._last_seen[fingerprint] = self._t
        if len(self._last_seen) > 2 * self.max_period:
            cutoff = self._t - self.max_period
            self._last_seen = {f: t for f, t in self._last_seen.items() if t >= cutoff}

        confirmed = (self._candidate
                     if self._candidate is not None
                     and self._streak >= self.min_periods * self._candidate
                     else None)
        if confirmed != self.period:
            # a period that stops holding is retracted, and its stationarity streak with it
            self.period, self._stationary, self.j_drift = confirmed, 0, None

        if self.period is not None and self._Js.has_lag(self.period):
            self.j_drift = abs(J - self._Js.lag(self.period))
            self._stationary = self._stationary + 1 if self.j_drift < self.j_eps else 0
        else:
            self._stationary = 0

        return self.cycle_period is not None and self._stationary >= self.period

    @property
    def cycle_period(self):
        """The confirmed period, but only where it describes an actual cycle.

        A converged cell repeats its assignment every iteration, so `period` is legitimately
        1 there -- a fixed point is a period-1 orbit. That is not what any reader of this
        value wants to know (stop_reason == "diff" already says it), and reporting it would
        make "cells with a confirmed cycle" count the whole diff branch. Hence period >= 2,
        the same bar the stopping rule uses.
        """
        return self.period if self.period is not None and self.period >= 2 else None


def threevarBAiterations(card_model, beta1, beta2, beta3, U_pre, pw, eps_conv, maxiter,
                         init_pogw_sparse=True, init_pogw_uniformly=False, init_pagow_uniformly=True,
                         init_pogw_blocks=False, init_state=None, verbose=True):
    """`init_state`, when given, is a full (po, pa, pogw, pago, pagow) tuple to start from and it
    overrides every init_pogw_* flag. It exists so a caller can start the solver *at* a named
    policy -- e.g. two_state_attention's top-2 imitator -- and ask whether the iteration stays
    there. Nothing else about the solver changes, and the flags behave exactly as before when it
    is None."""
    num_acts, num_worldstates = U_pre.shape
    history = []
    
    # damping coefficient for under-relaxation of the coupling marginals (po, pa, pago)
    alpha = settings.ALPHA

    if init_state is not None:
        # Copied, not aliased: _ba_step returns fresh arrays but the caller's tuple would
        # otherwise be the object the first iteration reads, and callers reuse these policies.
        state = tuple(np.array(x, dtype=float) for x in init_state)
        po_new, pa_new, pogw_new, pago_new, pagow_new = state

    # [Initialization Logic Remains Identical to BA_jl.py]
    elif init_pogw_uniformly:
        pogw_new = np.ones((card_model, num_worldstates)) / card_model
    elif init_pogw_blocks:
        # w_j -> o_k by CONTIGUOUS blocks of the w axis, so p(w|o) genuinely differs across o.
        #
        # This is the only init here that breaks the degeneracy at beta3 = 0. The sparse init
        # assigns by index parity (w_j -> o_(j mod card_model)), and the random init assigns
        # independently of w, so under both of them every model state sees essentially the same
        # p(w|o) -- measured on config 3 / card_model 3 / n 1000, E[w|o] differs by 0.11 across
        # a 228-wide range, and the resulting p(a|o) columns differ by 3.6e-3 against a 0.7
        # within-column spread. At beta3 = 0 that is fatal, because
        #
        #     pogw_util[k, j] = p(a|o=k) . U[:, j] - (1/beta2) D_KL( p(a|o=k) || p(a) )
        #
        # is constant in k once the p(a|o) columns coincide, giving p(o|w) = p(o), hence
        # p(w|o) = p(w), hence coinciding p(a|o) again. {p(a|o) columns identical} is an exactly
        # invariant AND attracting set: from the sparse init the column spread is 1.4e-16 after
        # one iteration and decays to 1.8e-25 by iteration 400, and the solver converges happily
        # onto it -- an effective cardinality of 1, only ever picking the single action whose
        # utility is best under p(w). At beta3 > 0 the same init escapes (spread 0.66 by
        # iteration 400 at beta3 = 0.5), which is why this only bites the serial case.
        #
        # Measured cost on a 5x4 (beta1, beta2) grid at beta3 = 0: the sparse init collapses to
        # cardinality 1 on 16 of 20 cells with J = -6.033 where this init reaches J = -5.25 to
        # -5.47, and it is never better on the remaining 4 (identical J to 1e-4).
        edges = (np.arange(num_worldstates) * card_model) // num_worldstates
        pogw_new = np.zeros((card_model, num_worldstates))
        pogw_new[np.minimum(edges, card_model - 1), np.arange(num_worldstates)] = 1.0
        pogw_new = (pogw_new + 0.01) / np.sum(pogw_new + 0.01, axis=0)
    elif init_pogw_sparse:
        pogw_new = np.eye(card_model, num_worldstates) if num_worldstates <= card_model else np.hstack([np.eye(card_model)] * (num_worldstates // card_model + 1))[:, :num_worldstates]
        # noise = np.random.rand(*pogw_new.shape) * 0.01
        pogw_new = (pogw_new + 0.01) / np.sum(pogw_new + 0.01, axis=0)
        # pogw_new = (pogw_new + noise) / np.sum(pogw_new + noise, axis=0)
    else:
        pogw_new = np.random.rand(card_model, num_worldstates)
        pogw_new /= np.sum(pogw_new, axis=0)

    if init_state is None:
        pagow_new = np.ones((num_acts, card_model, num_worldstates)) / num_acts

        po_new, pa_new, pagw = compute_marginals(pw, pogw_new, pagow_new)

        pago_new = compute_pago_iteration(pogw_new, pagow_new, beta2, beta3, U_pre, pa_new, po_new, pw)

        if beta3 == 0:
            # sequential case: force p(a|o,w) = p(a|o) so D_KL(p(a|o,w)||p(a|o)) = 0
            pagow_new = compute_pagow_iteration(pago_new, beta2, beta3, U_pre, pa_new)

        state = (po_new, pa_new, pogw_new, pago_new, pagow_new)

    # In the beta3 > beta2 regime, the hard (beta1=inf) p(o|w) assignment can
    # perpetually flip between tied options at ambiguous stimuli, so raw parameter
    # diff never settles. Track the objective value instead, compared across
    # J_CONV_WINDOW iterations (not consecutive ones -- see settings.py for why).
    case_specific_convergence = beta3 > beta2
    # Two adjacent J_CONV_WINDOW-length blocks: the criterion compares their means,
    # so the window has to hold both (see settings.py for why a two-sample J[t] vs
    # J[t-W] test cannot decay under a limit cycle).
    J_window = deque(maxlen=2 * settings.J_CONV_WINDOW)
    J_conv_streak = 0  # consecutive iterations with diff_J < J_EPS_CONV
    diff_conv_streak = 0  # consecutive iterations with diff < eps_conv
    # Runs in both branches. A fixed point is a period-1 cycle and belongs to the diff
    # criterion, so the detector only ever fires on period >= 2 (see _CycleDetector).
    detector = _CycleDetector()

    stop_reason = "maxiter"
    start_time = time.time()

    for i in range(maxiter):
        state, pagw, diff = _ba_step(state, beta1, beta2, beta3, U_pre, pw, alpha)

        # Scalar metrics only, no per-iteration matrix copies -- nothing downstream reads
        # the trajectory of the matrices themselves, only these columns of perf_df. The row
        # describes the state as it is AFTER this iteration's update, so it lines up with
        # the `diff` that produced it and the last row describes the state returned.
        row = _state_metrics(state, pagw, U_pre, pw, beta1, beta2, beta3)
        row["diff"] = diff
        J = row["Objective_value"]
        history.append(row)

        # --- rule 1: the parameter-diff criterion (a genuine fixed point) ---
        # a lone iteration below eps_conv proves nothing while the beta1=inf argmax
        # is still flipping -- one quiet iteration between flips passes the test at a
        # point that is not a fixed point (see settings.py). Require a sustained run.
        if diff < eps_conv:
            diff_conv_streak += 1
        else:
            diff_conv_streak = 0

        # --- rule 2: the limit-cycle criterion ---
        cycle_stationary = detector.update(_assignment_fingerprint(state[2]), J)

        # --- rule 3: the windowed-J criterion (beta3 > beta2 only) ---
        diff_J = None
        if case_specific_convergence:
            J_window.append(J)
            # only meaningful once both blocks are full. The mean of the most recent
            # J_CONV_WINDOW values against the mean of the J_CONV_WINDOW before them:
            # averaging cancels the limit cycle's oscillation, so what survives is the
            # drift -- the "still slowly moving" vs "actually done" distinction that a
            # consecutive-step diff (tiny by construction under this ALPHA) misses and
            # that a two-sample J[t] - J[t-W] comparison cannot resolve at all.
            if len(J_window) == J_window.maxlen:
                w = settings.J_CONV_WINDOW
                block = np.fromiter(J_window, dtype=float, count=2 * w)
                diff_J = abs(block[w:].mean() - block[:w].mean())
            row["diff_J"] = diff_J
            # a single dip below threshold can happen by chance mid-transient
            # (observed runs of up to ~60 consecutive spurious dips); require a
            # sustained streak so this only fires once J has actually settled
            if diff_J is not None and diff_J < settings.J_EPS_CONV:
                J_conv_streak += 1
            else:
                J_conv_streak = 0

        if verbose and (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            # detector.cycle_period, not detector.period: a converging cell repeats its
            # assignment every iteration, and printing "P=1" for that would say "cycle"
            # where the truth is "fixed point"
            cycle_str = ("none" if detector.cycle_period is None
                         else f"P={detector.cycle_period} drift={detector.j_drift:.1e}")
            if case_specific_convergence:
                diff_J_str = f"{diff_J:.3e}" if diff_J is not None else "n/a"
                print(f"[BA] iter {i + 1}/{maxiter}  diff={diff:.3e}  diff_J={diff_J_str} (window={settings.J_CONV_WINDOW})  streak={J_conv_streak}/{settings.J_CONV_PATIENCE}  (J_eps_conv={settings.J_EPS_CONV:.1e})  cycle={cycle_str}  elapsed={elapsed:.1f}s")
            else:
                print(f"[BA] iter {i + 1}/{maxiter}  diff={diff:.3e}  streak={diff_conv_streak}/{settings.DIFF_CONV_PATIENCE}  (eps_conv={eps_conv:.1e})  cycle={cycle_str}  elapsed={elapsed:.1f}s")

        # Priority: a fixed point is the strongest statement available, then a settled
        # cycle, then "J has stopped drifting" -- which says the trajectory has converged
        # in distribution but not which point of it to report.
        if diff_conv_streak >= settings.DIFF_CONV_PATIENCE:
            stop_reason = "diff"
            break
        if cycle_stationary:
            stop_reason = "limit_cycle"
            break
        if case_specific_convergence and J_conv_streak >= settings.J_CONV_PATIENCE:
            stop_reason = "windowed_J"
            break

    # --- best-J phase of the cycle ---
    # The orbit is exactly periodic (verified to machine precision on 15 of 16 sampled
    # cells), so the next `period` states are precisely its phases and this enumerates all
    # of them. J varies across those phases by up to 2.24 against a grid resolution of
    # 2.3e-3, so stopping on an arbitrary one is the difference between a solution and a
    # snapshot. The exit state is the incumbent, which also keeps this safe if the orbit
    # ever turns out to be only approximately periodic.
    best_J, best_state, best_pagw = history[-1]["Objective_value"], state, pagw
    if stop_reason == "limit_cycle":
        for _ in range(detector.period):
            state, pagw, diff = _ba_step(state, beta1, beta2, beta3, U_pre, pw, alpha)
            row = _state_metrics(state, pagw, U_pre, pw, beta1, beta2, beta3)
            row["diff"] = diff
            history.append(row)
            if row["Objective_value"] > best_J:
                best_J, best_state, best_pagw = row["Objective_value"], state, pagw

    # history rows are already scalar dicts matching the perf_df schema
    perf_df = pd.DataFrame(history)

    stop_info = {
        "stop_reason": stop_reason,
        "n_iters": len(history),
        # J of the state actually returned. Not perf_df["Objective_value"].iloc[-1]: on a
        # limit-cycle stop the last row is the end of the replay, not the phase chosen.
        "objective": float(best_J),
        # Recorded whenever a cycle was confirmed, whatever the solver stopped on -- a
        # `maxiter` row carrying a period is a cell that cycled but never settled. None on
        # the diff branch, where the "period" is 1 and the fixed point is the finding.
        "cycle_period": detector.cycle_period,
        "cycle_J_drift": detector.j_drift if detector.cycle_period else None,
        "diff_final": float(history[-1]["diff"]),
    }

    po_final, pa_final, pogw_final, pago_final, pagow_final = best_state
    return po_final, pa_final, pogw_final, pago_final, pagow_final, best_pagw, perf_df, stop_info
