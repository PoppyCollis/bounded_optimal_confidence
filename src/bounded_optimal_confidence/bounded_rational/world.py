"""One world per stimulus configuration (raw or normalised), and the operation that moves a router between two grids.

From ITBR_confidence/generalisation/world.py. `build` is cached per process. The stimulus means come from
`experiment.CATEGORY_X` (the original looked them up in the human-data fitting config; same values).
"""
import functools

import numpy as np

from bounded_optimal_confidence.experiment import CATEGORY_X
from bounded_optimal_confidence.bounded_rational import identity, settings
from bounded_optimal_confidence.bounded_rational import seeds as init_policy
from bounded_optimal_confidence.bounded_rational import world_model as norm_world


@functools.lru_cache(maxsize=64)
def build(stimulus_config, world=None, epsilon=None, sigma=None, n_samples=None, card_model=None,
          pw_prior=None, eps_sd=None):
    """(world_d, U_pre, pw) for one stimulus config. Cached per process, so a worker builds it once.

    Cached on the scalars alone, which is safe because every one of them is an argument with an
    explicit default resolved at call time -- nothing here reads a module global that a parent
    could have overridden after the child imported it.

    The dict carries everything `init_policy.init_states` and `identity.chirality` need on top of
    the arrays: the w grid, the stimulus means, the first Bayes boundary and whether this world is
    mirror-symmetric (configs 1 and 2 are; 3 and 4 are each other's mirror but neither is its own).

    `eps_sd` is the normalised world's own grid margin, in units of sig_p -- the knob `epsilon` is
    for the raw world. It is a separate argument rather than a reuse of `epsilon` because the two
    are in different units (absolute w against multiples of sig_p) and defaulting one from the
    other would silently rescale the grid. None keeps `norm_world.EPS_SD` (6), and it is ignored by
    the raw branch.
    """
    world = settings.WORLD if world is None else world
    epsilon = settings.EPSILON if epsilon is None else float(epsilon)
    sigma = settings.SIGMA if sigma is None else float(sigma)
    n_samples = settings.N_SAMPLES if n_samples is None else int(n_samples)
    card_model = settings.CARD_MODEL if card_model is None else int(card_model)
    pw_prior = settings.PW_PRIOR if pw_prior is None else pw_prior

    mu_as = np.asarray(mu_as_for(stimulus_config), dtype=float)
    if world == "raw":
        wd = norm_world.build("raw_uniform", mu_as, epsilon=epsilon, sigma=sigma,
                              n_samples=n_samples, pw_prior=pw_prior)
    else:
        # The normalised world sizes its own grid (eps_sd sig_p past the outer means at a fixed
        # step), so EPSILON and N_SAMPLES do not apply to it and are deliberately not passed.
        # eps_sd IS passed, because it is that world's own margin and the caller may want to shrink
        # it -- at eps_sd = 0 the grid runs from mu_1 to mu_A with no margin at all, which makes the
        # outer categories half-Gaussians once sum_w q(w|a) = 1 renormalises them. That is a real
        # world to ask about, so it is reachable; `norm_world.in_grid_mass` is how far from 1 it is.
        norm_kw = {} if eps_sd is None else {"eps_sd": float(eps_sd)}
        wd = norm_world.build("norm_mixture", mu_as, sig_s=sigma, **norm_kw)

    U_pre, pw, w_vals = wd["U_pre"], wd["pw"], wd["w_vals"]
    world_d = {
        "stimulus_config": int(stimulus_config),
        "world": world,
        "card_model": card_model,
        "epsilon": float(wd["epsilon"]),
        "sigma": float(wd["sig_s"]),
        "n_samples": int(wd["n_samples"]),
        "grid_step": float(wd["grid_step"]),
        "mu_as": mu_as,
        "w_vals": w_vals,
        "boundary": float(abs(init_policy.decision_boundaries(mu_as)[0])),
        "exclusion_cut": float(init_policy.exclusion_cut(mu_as)),
        "mirror_symmetric": identity.is_mirror_symmetric(mu_as),
    }
    return world_d, U_pre, pw


def mu_as_for(stimulus_config):
    """MU_AS for a numbered config: configs 1-4 of Li & Ma (2020) experiment 1 (`experiment.CATEGORY_X`)."""
    return np.asarray(CATEGORY_X[int(stimulus_config)], dtype=float)


def same_grid(w_a, w_b):
    """Do these two grids match exactly? Configs 1/3/4 do; config 2 does not."""
    w_a, w_b = np.asarray(w_a), np.asarray(w_b)
    return w_a.shape == w_b.shape and np.array_equal(w_a, w_b)


def regrid_pogw(pogw, w_src, w_tgt):
    """A router p(o|w) moved from one w grid to another by linear interpolation, edges held.

    "Edges held" is the np.clip: a target w outside the source grid takes the nearest source
    value rather than extrapolating. That is the behaviour the 2026-09-10 run used for config 2,
    whose grid extends 32 units further out on each side than the others, and it is the right one
    -- p(o|w) has already saturated well before the grid ends, so the held value IS the limit.

    Columns are renormalised after interpolation. They cannot drift far (interpolation of a
    convex combination stays one), but a router that sums to 1 - 2e-16 would put that error
    straight into true_terms' chain-rule assert.

    ON A HARD ROUTER. At beta1 = inf p(o|w) is a 0/1 step, and no interpolation reproduces a step.
    Measured on the cfg1 -> cfg2 -> cfg1 round trip: exactly 2 cells change, one either side of the
    cut, by up to 0.083 -- and the implied cut position moves by 0.000 grid steps. So the damage is
    confined to the discontinuity and does not displace it. At the finite beta1 this package sweeps
    the router is a soft sigmoid and the whole round trip is 7e-4 (softness 2) or 3e-5 (softness
    10). Neither case is approximated silently: the scores table carries a `regridded` flag per row,
    and test_agreement.check_regrid_identity asserts both bounds.
    """
    pogw = np.asarray(pogw, dtype=float)
    w_src, w_tgt = np.asarray(w_src, dtype=float), np.asarray(w_tgt, dtype=float)
    if same_grid(w_src, w_tgt):
        return pogw
    xf = np.clip(w_tgt, w_src[0], w_src[-1])
    out = np.stack([np.interp(xf, w_src, row) for row in pogw], axis=0)
    out = np.clip(out, 0.0, None)
    total = out.sum(axis=0, keepdims=True)
    return out / np.where(total > 0, total, 1.0)


def describe(world_d):
    """The world as log lines, for a run's config_log.txt."""
    mu = [float(m) for m in world_d["mu_as"]]
    w = world_d["w_vals"]
    return [f"cfg {world_d['stimulus_config']}  mu {mu}  "
            f"grid [{w[0]:g}, {w[-1]:g}] x {world_d['n_samples']} (step {world_d['grid_step']:.4f})"
            f"  cut {world_d['exclusion_cut']:g}  |d1| {world_d['boundary']:.4g}"
            f"  mirror-symmetric {world_d['mirror_symmetric']}"]
