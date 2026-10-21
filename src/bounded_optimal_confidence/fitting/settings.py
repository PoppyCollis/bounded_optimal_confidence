"""Every grid and constant of the fit, in one place.

The fit searches (sig_x, log10 alpha[, log10 beta]) on a grid; at every grid cell the b-triple and lapse are
profiled out (`likelihood.profile`). See docs/fitting.md for the procedure.
"""
import numpy as np

SEED = 0

# --- Monte Carlo budget (replicates per trial) --------------------------------------------------------------------
SIMNTRL = 10_000                # the refit of the shortlisted cells
RANK_SIMNTRL_FACTOR = 0.25      # the grid search: 2,500
CLEAN_SIMNTRL_FACTOR = 10       # the reported (clean) NLL: 100,000

# --- the grid -------------------------------------------------------------------------------------------------------
# sensory noise: sig_x = 0 exactly, then log-spaced 0.1-316 (36 rungs); a cell's index on this grid is `sig_x_index`
SIG_X_GRID = np.concatenate([[0.0], 10.0 ** np.linspace(-1.0, 2.5, 36)])
# Dirichlet decision noise: log10 alpha -0.5..4 at step 0.25, and alpha = inf (no decision noise)
LOG10_ALPHA_GRID = np.concatenate([np.linspace(-0.5, 4.0, 19), [np.inf]])

# --- coarse-to-fine search of the soft-attention models (the third axis, log10 beta) --------------------------------
# stage A: every SIG_X_STRIDE-th sig_x x LOG10_ALPHA_COARSE x LOG10_BETA_COARSE
LOG10_ALPHA_COARSE = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, np.inf])
# beta = 1 (the Bayes mixture) is interior; beta = inf (hard attention) is on the grid
LOG10_BETA_COARSE = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, np.inf])
SIG_X_STRIDE = 2
# stage B: around each subject's N_SEED best stage-A cells, +-SIG_X_STRIDE sig_x rungs x the fine alphas x the
# fine betas below
N_SEED = 3
FINE_STEP = 0.125
ALPHA_FINE_HALF = 0.5
BETA_FINE_HALF = 0.25
LOG10_BETA_LO, LOG10_BETA_HI = -1.25, 2.25      # the finite range stage B can reach

# --- clean stage ----------------------------------------------------------------------------------------------------
N_REFINE = 20                   # cells per subject re-profiled and clean-scored on fresh draws

# --- smoke runs (--smoke): a quick end-to-end check, not a fit --------------------------------------------------------
SMOKE_SIG_X_STRIDE = 4
SMOKE_N_REFINE = 3

# a grid edge (or a nested value of beta) within this many nats of a subject's best search cell is flagged
EDGE_TOL_NATS = 2.0


def _steps(lo, hi):
    """lo, lo + FINE_STEP, ... up to hi, rounded to 6 decimals."""
    return list(np.round(np.arange(lo, hi + 1e-9, FINE_STEP), 6))


def fine_log10_alphas(log10_alpha):
    """Stage B's log10 alphas around a stage-A value: +-ALPHA_FINE_HALF at FINE_STEP; around inf, the top of the
    finite grid and inf."""
    if np.isinf(log10_alpha):
        return [3.5, 3.75, 4.0, np.inf]
    return _steps(log10_alpha - ALPHA_FINE_HALF, log10_alpha + ALPHA_FINE_HALF)


def fine_log10_betas(log10_beta):
    """Stage B's log10 betas around a stage-A value: +-BETA_FINE_HALF at FINE_STEP within [LOG10_BETA_LO,
    LOG10_BETA_HI]. inf (hard attention) stays reachable: around inf, the top of the finite range and inf; from any
    finite value within a coarse step of it, inf is added."""
    top = [float(u) for u in _steps(LOG10_BETA_HI - 2 * BETA_FINE_HALF, LOG10_BETA_HI)]
    if np.isinf(log10_beta):
        return top + [np.inf]
    lo, hi = max(log10_beta - BETA_FINE_HALF, LOG10_BETA_LO), min(log10_beta + BETA_FINE_HALF, LOG10_BETA_HI)
    out = [float(u) for u in _steps(lo, hi)]
    return out + ([np.inf] if log10_beta >= LOG10_BETA_COARSE[-2] - 1e-9 else [])
