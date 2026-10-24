"""The constants of the bounded-rational hierarchy solve and of the worlds it is solved in.

Consolidated from the original repository's config modules (the root `config.py`, `partition_enum/config.py`,
`fixed_points/config.py`, `generalisation/config.py` and `beta_sweep/norm_world.py`); the values are theirs. Only the
constants the rate-utility sweep reaches are here. The original modules also took overrides from environment
variables; the sweep passes every world argument explicitly per task, so those never reached a number and are gone.
"""
import numpy as np

# --- The full three-variable solver (`blahut_arimoto.threevarBAiterations`) ------------------------------------
# Under-relaxation of po, pa and p(a|o). x and x = (1-a)x + aF(x) have the same fixed points for any a > 0, so damping
# cannot create or destroy solutions; it sets the path. 0.05 reaches the same solution as 0.001 (max |dp(a|o)| 1e-6,
# dJ 2e-7) in 326 iterations instead of 13888. Undamped, the iteration diverges.
ALPHA = 0.05
# `diff` is a per-iteration step size, not a distance to the fixed point, and with the hard (beta1 = inf) p(o|w)
# argmax one quiet iteration between assignment flips passes the test. So the residual has to stay below eps_conv for
# this many iterations in a row.
DIFF_CONV_PATIENCE = 300
# The windowed-J criterion applies only when beta3 > beta2 (never in the paper's sweep, where beta2 = inf): the
# means of two adjacent J_CONV_WINDOW-length blocks must differ by less than J_EPS_CONV for J_CONV_PATIENCE iterations.
J_EPS_CONV = 1e-4
J_CONV_WINDOW = 200
J_CONV_PATIENCE = 300
# Limit-cycle detection at beta1 = inf: the longest period the detector can represent, how many consecutive full
# periods confirm one, and the per-phase J drift below which the cycle counts as settled (settled orbits sit at
# exactly 0.0, unsettled ones at >= 3e-5).
CYCLE_MAX_PERIOD = 4000
CYCLE_MIN_PERIODS = 3
CYCLE_J_EPS = 1e-9
# Two p(a|o) columns closer than this (max abs) count as the same behaviour in the effective cardinality.
CARD_TOL = 5e-2

# --- The reduced solve that completes a seed router (`reduced.solve_reduced`) -------------------------------------
# With p(o|w) pinned the map is smooth, so it runs undamped and the plain diff criterion means what it says. The cap is
# 50000 rather than the enumeration's 5000 because at 5000 it bit on slow-but-converging cuts.
REDUCED_ALPHA = 1.0
REDUCED_EPS_CONV = 1e-8
REDUCED_DIFF_PATIENCE = 3
REDUCED_MAXITER = 50000

# --- Chirality of a solved state on a mirror-symmetric world (`identity.chirality`) -------------------------------
# |skew| below this is a self-mirror state (measured: genuine leans >= 1.4e-3, residues <= 2.5e-6). The fallback
# statistic is the crossing centroid, in units of the grid step.
CHIRALITY_SKEW_TOL = 1e-4
CHIRALITY_CENTROID_GRID_FRAC = 0.75

# --- The worlds (`world.build`) -----------------------------------------------------------------------------------
# Raw world: U(a, w) = log N(w; mu_a, SIGMA) on linspace(mu_1 - EPSILON, mu_A + EPSILON, N_SAMPLES), p(w) uniform.
EPSILON = 64.0
SIGMA = 64.0
N_SAMPLES = 1000
PW_PRIORS = ("uniform", "mixture")
PW_PRIOR = "uniform"
# Normalised world: U = log q(w|a) with q normalised over the grid, p(w) the equal-weight mixture, the grid EPS_SD
# sig_p past the outer means at the raw world's step (mu range + 2 * 64 over 999 intervals).
SIG_S = 64.0
EPS_SD = 6.0
GRID_STEP = (96.0 - -96.0 + 2 * 64.0) / 999.0
WORLDS = ("raw", "norm")
WORLD = "raw"           # the default world of `world.build` when none is named; the sweep always names one
# Two model states: the hierarchy of the paper.
CARD_MODEL = 2
# The four stimulus layouts of Li & Ma (2020) experiment 1: 1 = [-96, 0, 96], 2 = [-128, 0, 128], 3 = [-96, -59, 96],
# 4 = [-96, 59, 96]. Every one has mu_1 = -mu_3, so the top-2 exclusion cut is w = 0 on all four.
CONFIGS = (1, 2, 3, 4)

# --- The specialist solve (`candidates.solve_specialist`) ---------------------------------------------------------
# eps_conv 1e-8 rather than the original sweeps' 1e-6, because the mirror ties sit at ~1e-15 in J; the 50000 cap is
# sized for cells that start from a structured router.
EPS_CONV = REDUCED_EPS_CONV
FULL_MAXITER = 50000

# --- Run plumbing ---------------------------------------------------------------------------------------------------
N_WORKERS = 14


def config_lines():
    """Every public constant as 'NAME = value', for a run's config_log.txt."""
    return [f"{k} = {v!r}" for k, v in sorted(globals().items())
            if k.isupper() and not k.startswith("_")]
