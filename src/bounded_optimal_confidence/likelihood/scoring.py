"""The likelihood of a subject's (decision, confidence) reports under a simulated model.

A simulation gives, per trial, n_reps decisions d and continuous confidences c. The confidence boundaries
b1 <= b2 <= b3 bin c into the reported level r (np.digitize, right=False: level k holds b_k <= c < b_(k+1), the
MATLAB histc convention of the original code). Per trial, p(d, r) is the fraction of replicates in each cell, mixed
with a uniform lapse over all (d, r) cells and floored at EPS:

    p(d, r) = (1 - lapse) * count(d, r) / n_reps + lapse / (K * n_levels)

The NLL is -SUM_trials log p(d_obs, r_obs).
"""
import numpy as np

from bounded_optimal_confidence.experiment import N_CATEGORIES, N_CONFIDENCE_LEVELS
from bounded_optimal_confidence.models.simulate import CHUNK_REPS

EPS = np.finfo(float).eps       # probability floor


def report_joint_probs(sim_d, sim_conf, edges, lapse, ncat=N_CATEGORIES, nr=N_CONFIDENCE_LEVELS):
    """p(d, r | trial), shape (n_trials, ncat, nr), from a simulation (sim_d, sim_conf), each (n_reps, n_trials).

    `edges` is the sorted b-triple (nr - 1 boundaries). The lapse is mixed in before the EPS floor.
    """
    n_reps, ntrl = sim_d.shape
    sim_r = np.clip(np.digitize(sim_conf, edges, right=False), 0, nr - 1)
    trial_idx = np.broadcast_to(np.arange(ntrl), sim_d.shape)
    flat_idx = (trial_idx * ncat + sim_d) * nr + sim_r
    joint = np.bincount(flat_idx.ravel(), minlength=ntrl * ncat * nr).reshape(ntrl, ncat, nr)
    p_joint = (1 - lapse) * (joint / n_reps) + lapse * (1.0 / (ncat * nr))
    return np.maximum(p_joint, EPS)


def clean_nll(simulate_chunk, edges, lapse, decisions, confidences, n_reps, ncat=N_CATEGORIES,
              nr=N_CONFIDENCE_LEVELS):
    """The NLL at (edges, lapse) from `n_reps` replicates, simulated and counted CHUNK_REPS at a time.

    `simulate_chunk(n)` returns the next n replicates as (sim_d, sim_conf). Counts add over replicates, so this is
    exactly `report_joint_probs` on the whole simulation without holding it in memory. `decisions` and
    `confidences` are the subject's 1-based reports.
    """
    ntrl = len(decisions)
    counts = np.zeros(ntrl * ncat * nr, dtype=np.int64)
    trial_idx = None
    for start in range(0, n_reps, CHUNK_REPS):
        n = min(CHUNK_REPS, n_reps - start)
        sim_d, sim_conf = simulate_chunk(n)
        sim_r = np.clip(np.digitize(sim_conf, edges, right=False), 0, nr - 1)
        if trial_idx is None or trial_idx.shape != sim_d.shape:
            trial_idx = np.broadcast_to(np.arange(ntrl), sim_d.shape)
        flat = (trial_idx * ncat + sim_d) * nr + sim_r
        counts += np.bincount(flat.ravel(), minlength=ntrl * ncat * nr)
    p = counts.reshape(ntrl, ncat, nr) / n_reps
    p = np.maximum((1 - lapse) * p + lapse / (ncat * nr), EPS)
    return float(-np.log(p[np.arange(ntrl), decisions - 1, confidences - 1]).sum())
