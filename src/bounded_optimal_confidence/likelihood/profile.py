"""Profile the confidence boundaries and the lapse out of the likelihood of one simulation.

Simulating a model is the expensive step; re-scoring the SAME simulation at another (b1, b2, b3, lapse) is cheap.
So at every searched (sig_x, alpha[, beta]) the model is simulated once and the b-triple and lapse are minimised
out (`profile_b_and_lapse`):

1. Seed: an exhaustive sweep over NB-level ordered b-triples x LAPSES, scored approximately from a cumulative
   histogram of confidence at 1/NFINE resolution (`seed_table`). Safe for ranking, not for descending.
2. Polish: Nelder-Mead from the seed (or from a caller's warm start `x0`, whichever scores better) against the
   EXACT objective (`exact_joint_scorer`: the same counts as `report_joint_probs`, by binary search).
3. Report: the optimum re-scored with `scoring.report_joint_probs` and split into a decision term and a
   confidence-given-decision term.

The polish fits one Monte Carlo realisation, so a profiled NLL sits slightly below a fresh draw's; the fit therefore
reports a clean NLL on an independent draw (`scoring.clean_nll`).
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, minimize_scalar

from bounded_optimal_confidence.experiment import N_CATEGORIES, N_CONFIDENCE_LEVELS
from bounded_optimal_confidence.likelihood.scoring import EPS, report_joint_probs

NFINE = 4096        # confidence bins of the seed's cumulative histogram (every readout is in [0, 1])
NB = 40             # b levels per axis for the seed sweep: 11,480 ordered triples
# lapse levels for the seed; deliberately reaching 0.8 so a high-lapse basin is found at all
LAPSES = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.80)
POLISH_MAXFEV = 400  # Nelder-Mead budget; ~160 evaluations is typical, the cap guards a degenerate simplex
POLISH_XATOL = 1e-4
POLISH_FATOL = 1e-3


@dataclass
class ProfileResult:
    """One simulation with the b-triple and lapse minimised out.

    nll_joint = nll_d + nll_rgd exactly (decision term + confidence-given-decision term at the one optimum).
    nll_d_min is the best the decision term alone could reach with its own lapse (the b-triple does not enter it).
    """

    nll_joint: float
    nll_d: float
    nll_rgd: float
    nll_d_min: float
    edges: np.ndarray       # the sorted b-triple attaining nll_joint
    lapse: float            # the lapse attaining nll_joint
    lapse_d_min: float      # the lapse attaining nll_d_min
    seed_nll: float         # the seed sweep's (approximate) value before polishing
    n_polish_evals: int
    warm_start_used: bool = False   # True when the caller's x0 beat the seed and started the simplex


def b_triple_grid(nb=NB, nfine=NFINE):
    """Every ordered b1 <= b2 <= b3 on an nb-level grid of [0, 1], as fine-bin indices (n_triples, 3)."""
    levels = np.linspace(0.0, 1.0, nb)
    i, j, k = np.meshgrid(np.arange(nb), np.arange(nb), np.arange(nb), indexing="ij")
    keep = (i <= j) & (j <= k)
    tri = np.stack([levels[i[keep]], levels[j[keep]], levels[k[keep]]], axis=1)
    return np.clip(np.round(tri * nfine).astype(int), 0, nfine)


_B_IDX_CACHE = {}


def _b_idx(nb, nfine):
    if (nb, nfine) not in _B_IDX_CACHE:
        _B_IDX_CACHE[nb, nfine] = b_triple_grid(nb, nfine)
    return _B_IDX_CACHE[nb, nfine]


def cumulative_counts(sim_d, sim_conf, d_obs, nfine=NFINE):
    """(n_trials, nfine + 1): entry [t, k] counts the replicates of trial t that chose d_obs[t] AND read out a
    confidence below fine-bin edge k. Column nfine is the total, the unnormalised model p(d_obs)."""
    n_reps, ntrl = sim_d.shape
    hit = sim_d == d_obs[None, :]
    kfine = np.clip((np.asarray(sim_conf) * nfine).astype(int), 0, nfine - 1)
    flat = np.arange(ntrl)[None, :] * nfine + kfine
    counts = np.bincount(flat[hit].ravel(), minlength=ntrl * nfine).reshape(ntrl, nfine)
    cum = np.zeros((ntrl, nfine + 1))
    cum[:, 1:] = np.cumsum(counts, axis=1)
    return cum


def seed_table(cum, r_obs, n_reps, ncat, nr, nb=NB, lapses=LAPSES, nfine=NFINE):
    """(len(lapses), n_triples) approximate joint NLL at every (lapse, b-triple) of the seed grid, and the triples'
    fine-bin indices."""
    b_idx = _b_idx(nb, nfine)
    ntrl = cum.shape[0]
    rows = np.arange(ntrl)[:, None]
    c1 = cum[rows, b_idx[:, 0][None, :]]          # (n_trials, n_triples)
    c2 = cum[rows, b_idx[:, 1][None, :]]
    c3 = cum[rows, b_idx[:, 2][None, :]]
    ctot = cum[:, nfine][:, None]
    # the subject's own report level picks which difference of cumulative columns counts the observed (d, r) cell
    cnt = np.where(r_obs[:, None] == 0, c1,
          np.where(r_obs[:, None] == 1, c2 - c1,
          np.where(r_obs[:, None] == 2, c3 - c2, ctot - c3)))
    q = cnt / n_reps                               # (n_trials, n_triples) model p(d_obs, r_obs)
    table = np.empty((len(lapses), len(b_idx)))
    for k, lapse in enumerate(lapses):
        p = np.maximum((1 - lapse) * q + lapse / (ncat * nr), EPS)
        table[k] = -np.log(p).sum(axis=0)
    return table, b_idx


def seed_over_b_and_lapse(cum, r_obs, n_reps, ncat, nr, nb=NB, lapses=LAPSES, nfine=NFINE):
    """Argmin of `seed_table`: (approximate nll, edges, lapse), the polish's starting point."""
    table, b_idx = seed_table(cum, r_obs, n_reps, ncat, nr, nb, lapses, nfine)
    best, best_edges, best_lapse = np.inf, None, np.nan
    for lapse, nll in zip(lapses, table):
        idx = int(np.argmin(nll))
        if nll[idx] < best:
            best = float(nll[idx])
            best_edges = b_idx[idx] / nfine
            best_lapse = float(lapse)
    return best, np.asarray(best_edges, dtype=float), best_lapse


def decompose(sim_d, sim_conf, edges, decisions, confidences, ncat, nr, lapse):
    """(nll_joint, nll_d, nll_rgd): -log p(d, r) split into -log p(d) and -log p(r | d), via report_joint_probs."""
    p_joint = report_joint_probs(sim_d, sim_conf, np.sort(edges), lapse, ncat, nr)
    t = np.arange(sim_d.shape[1])
    d_idx, r_idx = np.asarray(decisions) - 1, np.asarray(confidences) - 1
    nll_joint = float(-np.log(p_joint[t, d_idx, r_idx]).sum())
    nll_d = float(-np.log(p_joint[t, d_idx, :].sum(axis=1)).sum())
    return nll_joint, nll_d, nll_joint - nll_d


def exact_joint_scorer(sim_d, sim_conf, d_obs, r_obs, ncat, nr):
    """f(edges, lapse) -> the exact joint NLL (as report_joint_probs), ~100x cheaper than re-binning.

    The count of a trial's replicates below an edge is a binary search on that trial's sorted confidences. To keep
    the trials apart in one sorted array exactly, confidences are rank-transformed to integers and each trial is
    offset by an integer stride (a float offset would merge confidences piled up within 1e-16 of 1.0).
    """
    n_reps, ntrl = sim_d.shape
    conf = np.asarray(sim_conf)
    d_obs, r_obs = np.asarray(d_obs), np.asarray(r_obs)

    hit = sim_d == d_obs[None, :]
    trial_of = np.broadcast_to(np.arange(ntrl), sim_d.shape)[hit]
    values, rank = np.unique(conf[hit], return_inverse=True)
    stride = np.int64(len(values) + 1)
    key = np.sort(trial_of.astype(np.int64) * stride + rank.astype(np.int64))

    base = np.arange(ntrl, dtype=np.int64) * stride
    lo = np.searchsorted(key, base, side="left")
    total = np.searchsorted(key, base + stride, side="left") - lo

    def score(edges, lapse):
        # side="left" on `values` counts confidences strictly below the edge: digitize(right=False)'s [b_k, b_(k+1))
        c = [np.searchsorted(key, base + np.searchsorted(values, b, side="left"), side="left") - lo for b in edges]
        cnt = np.where(r_obs == 0, c[0],
              np.where(r_obs == 1, c[1] - c[0],
              np.where(r_obs == 2, c[2] - c[1], total - c[2])))
        p = np.maximum((1 - lapse) * cnt / n_reps + lapse / (ncat * nr), EPS)
        return float(-np.log(p).sum())

    return score


def decision_marginal_nll(q_d, ncat, lapse):
    """-SUM_t log p(d_obs | trial) from the per-trial hit rates q_d; the lapse marginalises to uniform over ncat."""
    return float(-np.log(np.maximum((1 - lapse) * q_d + lapse / ncat, EPS)).sum())


def best_decision_marginal(q_d, ncat):
    """min over lapse of `decision_marginal_nll` as (nll, lapse). Convex in lapse; the endpoints are checked
    explicitly because the bounded scalar minimiser never evaluates them (lapse = 0 is common)."""
    objective = lambda lapse: decision_marginal_nll(q_d, ncat, lapse)
    res = minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded")
    return min([(float(res.fun), float(res.x)), (objective(0.0), 0.0), (objective(1.0), 1.0)])


def profile_b_and_lapse(sim_d, sim_conf, decisions, confidences, x0=None, ncat=N_CATEGORIES, nr=N_CONFIDENCE_LEVELS,
                        nb=NB, lapses=LAPSES, nfine=NFINE):
    """Minimise the joint NLL over the b-triple and lapse for one simulation; returns a ProfileResult.

    `decisions` / `confidences` are the subject's 1-based reports. `x0` is an optional warm start (b1, b2, b3,
    lapse): it is scored exactly against the seed, and the better of the two starts the simplex.
    """
    d_obs = np.asarray(decisions) - 1
    r_obs = np.asarray(confidences) - 1
    n_reps = sim_d.shape[0]

    cum = cumulative_counts(sim_d, sim_conf, d_obs, nfine)
    seed_nll, seed_edges, seed_lapse = seed_over_b_and_lapse(cum, r_obs, n_reps, ncat, nr, nb, lapses, nfine)
    score = exact_joint_scorer(sim_d, sim_conf, d_obs, r_obs, ncat, nr)

    def objective(v):
        return score(np.sort(np.clip(v[:3], 0.0, 1.0)), float(np.clip(v[3], 0.0, 1.0)))

    start = np.concatenate([seed_edges, [seed_lapse]])
    # with no warm start the fallback is the seed's approximate value; with one, both origins are priced exactly
    fallback_nll = seed_nll
    warm_start_used = False
    if x0 is not None:
        cand = np.asarray(x0, dtype=float)
        f_seed, f_cand = objective(start), objective(cand)
        if f_cand < f_seed:
            start, fallback_nll, warm_start_used = cand, f_cand, True
        else:
            fallback_nll = f_seed

    res = minimize(objective, start, method="Nelder-Mead",
                   options=dict(xatol=POLISH_XATOL, fatol=POLISH_FATOL, maxfev=POLISH_MAXFEV))
    # Nelder-Mead is unconstrained and the objective clips, so report the clipped point
    edges = np.sort(np.clip(res.x[:3], 0.0, 1.0))
    lapse = float(np.clip(res.x[3], 0.0, 1.0))
    if res.fun > fallback_nll:
        edges, lapse = np.sort(np.clip(start[:3], 0.0, 1.0)), float(np.clip(start[3], 0.0, 1.0))

    nll_joint, nll_d, nll_rgd = decompose(sim_d, sim_conf, edges, decisions, confidences, ncat, nr, lapse)
    q_d = (sim_d == d_obs[None, :]).mean(axis=0)
    nll_d_min, lapse_d_min = best_decision_marginal(q_d, ncat)
    if decision_marginal_nll(q_d, ncat, lapse) < nll_d_min:      # keeps nll_d_min <= nll_d structural
        nll_d_min, lapse_d_min = decision_marginal_nll(q_d, ncat, lapse), lapse

    return ProfileResult(nll_joint=nll_joint, nll_d=nll_d, nll_rgd=nll_rgd, nll_d_min=nll_d_min, edges=edges,
                         lapse=lapse, lapse_d_min=lapse_d_min, seed_nll=seed_nll, n_polish_evals=int(res.nfev),
                         warm_start_used=warm_start_used)
