"""The work done in worker processes: score grid cells (search stage) and clean-score a shortlisted cell.

A pooled fit uses ONE parameter set for all four configs of a subject. Each config is simulated on its own trials,
and the simulations are joined along the trial axis (configs in order 1-4), so the b-triple and lapse are profiled
once on all of the subject's trials.
"""
from dataclasses import dataclass

import numpy as np
import threadpoolctl

from bounded_optimal_confidence.experiment import CONFIGS, N_CATEGORIES, load_subject
from bounded_optimal_confidence.fitting.seeding import clean_key, rank_rng, refit_rng, score_rng
from bounded_optimal_confidence.fitting.settings import SIG_X_GRID
from bounded_optimal_confidence.likelihood import clean_nll, profile_b_and_lapse
from bounded_optimal_confidence.models import get_model, simulate

# full-entropy confidence of a 50/50 two-category posterior: the lowest confidence hard attention can give
FULL_ENTROPY_FLOOR = 1.0 - 1.0 / np.log2(N_CATEGORIES)


@dataclass(frozen=True)
class SearchTask:
    """Score `cells` ((log10_alpha, log10_beta), ...) at one sig_x for one subject. log10_beta is None for the
    models without a beta."""

    model: str
    subject: int
    sig_x_index: int
    cells: tuple
    n_reps: int
    seed: int


@dataclass(frozen=True)
class CleanTask:
    """Re-profile one shortlisted cell on a fresh draw, then score it on another, larger one."""

    model: str
    subject: int
    sig_x_index: int
    log10_alpha: float
    log10_beta: float | None
    n_reps: int
    clean_n_reps: int
    seed: int


def pooled_reports(blocks):
    """The subject's (decisions, confidences) over all configs, in the order the simulations are joined."""
    return np.concatenate([b.decision for b in blocks]), np.concatenate([b.confidence for b in blocks])


def simulate_pooled(model, blocks, sig_x, log10_alpha, log10_beta, n_reps, rngs):
    """(decisions, confidences), each (n_reps, all trials): every config simulated with its own generator."""
    parts = [simulate(model, cfg, trials, sig_x, log10_alpha, log10_beta, n_reps, rng)
             for cfg, trials, rng in zip(CONFIGS, blocks, rngs)]
    return np.concatenate([p[0] for p in parts], axis=1), np.concatenate([p[1] for p in parts], axis=1)


def profile(model, sim_d, sim_conf, decisions, confidences):
    """`profile_b_and_lapse`, made safe for the soft-attention models.

    Their confidence (full entropy) is, near hard selection, the two-category scale squeezed onto
    [FULL_ENTROPY_FLOOR, 1], and the b-triple search (a seed grid on [0, 1], Nelder-Mead steps relative to the
    coordinates) is not scale-invariant: it can miss the optimum found on the two-category scale (by up to ~50 nats at
    a subject's best cell). So they are profiled on that scale first -- confidence mapped back to [0, 1], anything
    below the floor lumped at 0 -- and the result, mapped across, warm-starts the full-scale profile. A soft model
    therefore never scores worse than its hard twin at a shared cell.
    """
    if get_model(model).attention != "soft":
        return profile_b_and_lapse(sim_d, sim_conf, decisions, confidences)
    lo = FULL_ENTROPY_FLOOR
    two = profile_b_and_lapse(sim_d, np.clip((sim_conf - lo) / (1.0 - lo), 0.0, 1.0), decisions, confidences)
    x0 = np.concatenate([lo + (1.0 - lo) * np.asarray(two.edges[:3], dtype=float), [two.lapse]])
    return profile_b_and_lapse(sim_d, sim_conf, decisions, confidences, x0=x0)


def _cell_columns(model, log10_alpha, log10_beta):
    return {"log10_alpha": log10_alpha, **({"log10_beta": log10_beta} if get_model(model).fits_beta else {})}


def score_cells(task):
    """Search stage: one row per cell, with the profiled NLL and the b-triple and lapse that attain it."""
    sig_x = float(SIG_X_GRID[task.sig_x_index])
    rows = []
    with threadpoolctl.threadpool_limits(1):
        blocks = load_subject(task.subject)
        decisions, confidences = pooled_reports(blocks)
        for la, lb in task.cells:
            rngs = [rank_rng(task.seed, task.subject, cfg) for cfg in CONFIGS]
            sim = simulate_pooled(task.model, blocks, sig_x, la, lb, task.n_reps, rngs)
            res = profile(task.model, *sim, decisions, confidences)
            rows.append({"subject": task.subject, "sig_x_index": task.sig_x_index, "sig_x": sig_x,
                         **_cell_columns(task.model, la, lb), "profile_nll": res.nll_joint,
                         "b1": res.edges[0], "b2": res.edges[1], "b3": res.edges[2], "lapse": res.lapse})
    return rows


def clean_score(task):
    """Clean stage: refit the b-triple and lapse at `n_reps` on a fresh draw, then the NLL at those values from
    `clean_n_reps` replicates of another draw."""
    sig_x = float(SIG_X_GRID[task.sig_x_index])
    la, lb = task.log10_alpha, task.log10_beta
    key = clean_key(task.seed, task.subject, task.sig_x_index, la, lb)
    with threadpoolctl.threadpool_limits(1):
        blocks = load_subject(task.subject)
        decisions, confidences = pooled_reports(blocks)
        refit = [refit_rng(key, i) for i in range(len(blocks))]
        res = profile(task.model, *simulate_pooled(task.model, blocks, sig_x, la, lb, task.n_reps, refit),
                      decisions, confidences)
        edges = np.asarray(res.edges[:3], dtype=float)
        score = [score_rng(key, i) for i in range(len(blocks))]
        nll = clean_nll(lambda n: simulate_pooled(task.model, blocks, sig_x, la, lb, n, score), edges, res.lapse,
                        decisions, confidences, task.clean_n_reps)
    return {"subject": task.subject, "sig_x_index": task.sig_x_index, "sig_x": sig_x,
            **_cell_columns(task.model, la, lb), "refit_profile_nll": res.nll_joint, "nll": nll,
            "b1": edges[0], "b2": edges[1], "b3": edges[2], "lapse": res.lapse}
