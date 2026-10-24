"""The rate-utility curve of the top-2 code, and of the hierarchy that is solved for instead.

    python scripts/rate_utility_sweep.py                             # both worlds, cfgs 1-4
    python scripts/rate_utility_sweep.py --tag smoke \
            --configs 1 3 --worlds raw --beta3-n 6                   # shapes only, ~1 min
    python paper_plots/fig_rate_cost_utility.py                      # the figure

THE OBJECT. The top-2 code is EXPRESSIBLE in the 3-variable bounded-rational architecture without
being a fixed point of it: pin p(o|w) to a hard cut at w = 0 (beta1 = inf), pin
p(a|o) = [[q,0],[1-q,1-q],[0,q]] at q = 0.5 (beta2 = inf), and the only knob left is beta3, which
tempers the decision layer p(a|o,w) ~ p(a|o) exp(beta3 U). Sweeping beta3 traces a curve in the
(rate, utility) plane. This module traces that curve, and beside it the curve of what the
architecture actually SOLVES for at the same (inf, inf, beta3), on every stimulus layout.

WHY beta1 = beta2 = inf MAKES THE PICTURE READABLE. Both serial charges drop out of

    J = E[U] - I(W;O)/beta1 - I(A;O)/beta2 - I(W;A|O)/beta3

so J = E[U] - I(W;A|O)/beta3 exactly, for BOTH families. In the (I(W;A|O)/beta3, E[U]) plane that
is a 45-degree offset: the vertical gap between the two curves at a shared x IS the J gap, with
nothing to reconstruct per point. The (I(W;A|O), E[U]) plane is the same data at the natural rate
units, where the slope of the curve is 1/beta3 rather than 1.

ONE SCORING FUNCTION FOR BOTH FAMILIES. Everything goes through `transfer.score`, which builds its
information terms from the joint p(a,o,w) via `beta_sweep.info_terms.true_terms`. That matters
because one family is PINNED: on a pinned policy `policies.info_terms` reports the objective's
CHARGES, not mutual informations (I(W;A|O) = 0.185 against a true 0.075 on config 3), so scoring
the two families through different routes would put them in different accountings and the gap
between the curves would not mean anything. The solved family is re-scored through the same
function on its own config, and the self-transfer identity -- that re-scoring reproduces the J it
was solved at -- is recorded per row rather than assumed.

NO PLOTTING IMPORTS. Task functions run in spawned workers; pyplot costs ~150 MB per process and a
16-worker run in the original repo was OOM-killed for exactly this.

From ITBR_confidence/generalisation/rate_utility_beta3.py. The run folder goes under results/rate_utility/, the
config log dumps `settings` instead of the original's three config modules, and the git commit is this repository's.
"""
import argparse
import os
from collections import Counter
from concurrent.futures import as_completed
from datetime import datetime

import numpy as np
import pandas as pd
import threadpoolctl
from tqdm import tqdm

from bounded_optimal_confidence.paths import RATE_UTILITY_DIR
from bounded_optimal_confidence.utils.parallel import worker_pool
from bounded_optimal_confidence.utils.provenance import git_commit
from bounded_optimal_confidence.bounded_rational import candidates, settings, transfer
from bounded_optimal_confidence.bounded_rational import world as gworld
from bounded_optimal_confidence.bounded_rational.blahut_arimoto import compute_marginals, compute_pagow_iteration
from bounded_optimal_confidence.bounded_rational.info_terms import rate_utility, true_terms
from bounded_optimal_confidence.bounded_rational.rate_distortion import _collapsed_branch_applies, collapsed_branch, onevar_ba
from bounded_optimal_confidence.bounded_rational.reduced import marginalizeo_fast

OUTPUT_DIR = str(RATE_UTILITY_DIR)
LOG_NAME = "config_log.txt"


FAMILIES = ("top2", "solved", "rd")


BETA3_MARKS = (1.0,)


BETA3_MIN = 0.05


BETA3_MAX = 100.0


BETA3_N = 80


BETA1 = float("inf")


BETA2 = float("inf")


SELF_TRANSFER_LOUD = 1e-4


RUN_PREFIX = "rateutil"


CURVES_NAME = "curves.csv"


LATEST_NAME = "latest_rateutil.txt"


def beta3_axis(n=None, lo=None, hi=None, marks=BETA3_MARKS):
    """The beta3 axis, with `marks` folded in EXACTLY.

    Geometric, because the interesting structure is at small beta3 and the utility saturates at
    large. Each mark replaces its nearest rung rather than being appended, so the axis keeps its
    length and stays sorted -- the same trick `beta_sweep.mi_confidence_readout.beta_grid` uses.
    """
    n = BETA3_N if n is None else int(n)
    lo = BETA3_MIN if lo is None else float(lo)
    hi = BETA3_MAX if hi is None else float(hi)
    axis = np.geomspace(lo, hi, n)
    for m in marks:
        if lo <= m <= hi:
            axis[int(np.abs(np.log(axis) - np.log(m)).argmin())] = float(m)
    return np.unique(axis)


def _finite(row, keys):
    return all(np.isfinite(row[k]) for k in keys)


_TERM_KEYS = ("E_U", "J", "I_WO", "I_AO", "I_WAgO", "I_WA", "I_AOgW")


def _blank():
    """A row for a cell whose scoring blew up, with the flag that says so.

    Not a crash and not a silent drop: the beta3 at which a family stops being representable is
    itself a fact about that family, and a dropped row would hide where it is.
    """
    out = {k: float("nan") for k in _TERM_KEYS}
    out.update({"n_actions_used": -1, "regridded": False, "finite": False})
    return out


def lean_pagow(pogw, pago, beta2, beta3, U_pre, pw):
    """p(a|o,w) for a pinned pathway, with the dead cells of the router masked before p(a) is read.

    This is `policies.state_from` plus one `np.where`, and the `np.where` is why this exists.
    p(a|o,w) is built for EVERY model state at EVERY w, including the (o, w) pairs the router gives
    zero probability. Out in the tail, the state that has already discarded the only high-utility
    action has both of its surviving columns underflow, so that cell is 0/0 = NaN. It carries no
    weight -- p(o|w) is 0 there -- but `state_from` reads a marginal p(a) off the unmasked array
    and then re-runs the update with it, and a NaN marginal poisons every cell, so p(a|w) comes
    back NaN EVERYWHERE rather than only where the router is dead.

    Measured here: config 1, norm world, the whole policy is NaN at beta3 = 100. memory.md records
    the same failure at beta3 ~ 147 for `policies.top2_imitator` on config 3. So the cliff is
    config- and world-dependent and there is no safe fixed ceiling to sweep under.

    Masking first is exact, not a repair: `true_terms` and `marginalizeo_fast` both weight by
    p(o|w), so setting the dead cells to zero changes no quantity that is defined. `check_lean`
    asserts this against `transfer.score` on cells where that function still works.
    """
    card_a = U_pre.shape[0]
    live = np.asarray(pogw) > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        pagow = compute_pagow_iteration(pago, beta2, beta3, U_pre, np.ones(card_a) / card_a)
        pagow = np.where(live[None], pagow, 0.0)
        if not np.isinf(beta2):
            # At finite beta2 the placeholder p(a) above biases the update, so the pair is
            # iterated once more -- exactly what state_from does, and for the same reason.
            _po, pa, _ = compute_marginals(pw, pogw, pagow)
            pagow = np.where(live[None], compute_pagow_iteration(pago, beta2, beta3, U_pre, pa), 0.0)
    return pagow


def score_pathway(pogw, pago, w_src, stimulus_config, betas, world_kw):
    """E[U], J and the true information terms for a frozen pathway at one (config, beta3).

    `transfer.score` with `lean_pagow` in place of `policies.state_from`, and with the NaN cliff
    turned into a flag instead of an exception. Same accounting as the rest of the package: terms
    from the joint via `true_terms`, J via `transfer.objective`. `check_lean` pins the two
    together, so this cannot quietly become a second accounting.
    """
    beta1, beta2, beta3 = betas
    world_d, U_pre, pw = gworld.build(stimulus_config, **world_kw)
    w_tgt = world_d["w_vals"]
    regridded = not gworld.same_grid(w_src, w_tgt)
    pogw_t = gworld.regrid_pogw(pogw, w_src, w_tgt) if regridded else np.asarray(pogw, dtype=float)

    pagow = lean_pagow(pogw_t, pago, beta2, beta3, U_pre, pw)
    # Only the LIVE cells have to be finite. A dead cell is masked to 0 above and carries no
    # weight; a live cell that underflowed to 0/0 is a real failure of the pinned update and the
    # row says so rather than being renormalised into something plausible.
    if not np.isfinite(pagow[:, np.asarray(pogw_t) > 0]).all():
        return _blank()
    pagw = marginalizeo_fast(pogw_t, pagow)
    terms = true_terms(pw, pogw_t, pagow)
    eu, i_wa = rate_utility(pagw, U_pre, pw)
    assert abs(i_wa - terms["I(W;A)"]) < 1e-9, \
        f"I(W;A) routes disagree by {i_wa - terms['I(W;A)']:.2e} on cfg {stimulus_config}"
    row = {"E_U": float(eu), "J": transfer.objective(terms, eu, beta1, beta2, beta3),
           "I_WO": float(terms["I(W;O)"]), "I_AO": float(terms["I(A;O)"]),
           "I_WAgO": float(terms["I(W;A|O)"]), "I_WA": float(terms["I(W;A)"]),
           "I_AOgW": float(terms["I(A;O|W)"]),
           "n_actions_used": int((pagw.max(axis=1) > 1e-3).sum()), "regridded": bool(regridded)}
    row["finite"] = _finite(row, _TERM_KEYS)
    return row


def cell(task):
    """Every row for one (world, config, beta3). Top-level and picklable, for the worker pool."""
    world_kw, stimulus_config, beta3, q, betas_head, want_rd = task
    with threadpoolctl.threadpool_limits(1):
        beta1, beta2 = betas_head
        betas = (beta1, beta2, float(beta3))
        world_d, U_pre, pw = gworld.build(stimulus_config, **world_kw)
        common = {"world": world_kw["world"], "stimulus_config": int(stimulus_config),
                  "beta1": beta1, "beta2": beta2, "beta3": float(beta3), "q": float(q)}
        rows = []

        # --- the pinned top-2 code ---------------------------------------------------------
        # Rebuilt natively on this grid rather than transported: the exclusion cut is (mu_1+mu_A)/2
        # and every config has mu_1 = -mu_3, so it is w = 0 on all four and there is nothing
        # config-specific to carry over. beta1 never touches this router -- it is hard by
        # construction -- so it enters only through the (zero, at inf) I(W;O) charge.
        pogw, pago = candidates.top2_pathway(world_d, q)
        rows.append({**common, "family": "top2",
                     **score_pathway(pogw, pago, world_d["w_vals"], stimulus_config, betas,
                                     world_kw)})

        # --- the solved card-2 hierarchy ---------------------------------------------------
        spec = candidates.solve_specialist(stimulus_config, betas, world_kw=world_kw,
                                           include_collapsed=False)
        srow = score_pathway(spec["pogw"], spec["pago"], spec["w_vals"], stimulus_config, betas,
                             world_kw)
        resid = abs(srow["J"] - spec["J"]) if np.isfinite(srow["J"]) else float("nan")
        rows.append({**common, "family": "solved", **srow,
                     "self_transfer_resid": float(resid),
                     "J_solver": float(spec["J"]),
                     **{k: spec[k] for k in ("seed", "stop_reason", "n_iters", "diff_final",
                                             "effective_cardinality", "chirality",
                                             "J_spread_seeds")}})

        # --- references, not curves --------------------------------------------------------
        # The collapsed (p(o|w) = p(o)) branch is card 1. The question fixes the cardinality at 2,
        # so it is excluded from the solve above -- but a beta3 where it would have won is worth
        # knowing, and a column says so without a second family appearing mid-axis.
        if _collapsed_branch_applies(beta3):
            rows[-1]["J_collapsed"] = float(collapsed_branch(
                beta1, beta2, float(beta3), world_d["card_model"], U_pre, pw)[0])
        else:
            rows[-1]["J_collapsed"] = float("nan")

        if want_rd:
            # The 1-step rate-distortion frontier: no model state at all, so its rate is I(W;A)
            # and it has no I(W;A|O) to speak of. No policy's J beats it at its own price, which
            # is why it belongs on the figure as a ceiling. COLD-STARTED at every beta3 -- warm
            # starting onevar_ba is 17x faster and lands on a different fixed point, and an
            # argument for it has been written and reverted in this repo once already.
            _pa, pagw, _n = onevar_ba(float(beta3), U_pre, pw)
            eu, i_wa = rate_utility(pagw, U_pre, pw)
            rows.append({**common, "family": "rd", "E_U": float(eu),
                         "J": float(eu - i_wa / float(beta3)),
                         "I_WO": 0.0, "I_AO": 0.0, "I_WAgO": float(i_wa), "I_WA": float(i_wa),
                         "I_AOgW": 0.0, "n_actions_used": int((pagw.max(axis=1) > 1e-3).sum()),
                         "regridded": False, "finite": bool(np.isfinite(eu) and np.isfinite(i_wa))})

        for r in rows:
            b3 = r["beta3"]
            r["cost_cond"] = r["I_WAgO"] / b3
            r["cost_total"] = r["I_WA"] / b3
        return rows


def check_hard_router(curves, world_kws, tol=1e-9):
    """On a hard cut I(A;O|W) = 0 and I(W;O) = H(p(o)), so I(W;A) = I(A;O) + I(W;A|O) exactly.

    This is the identity that makes the two rate columns of the figures two views of ONE
    decomposition rather than two unrelated x axes, so it is measured rather than asserted from the
    armchair. It holds for the top-2 family only: a solved router is soft and carries its own noise
    into A, which is what I(A;O|W) > 0 means.

    H(p(o)), NOT log 2. A deterministic router's I(W;O) is the entropy of the side it lands on,
    and that is log 2 only when the cut splits p(w) in half. It does on the raw world, whose p(w)
    is uniform on a grid symmetric about the cut, and on the symmetric configs. It does NOT in the
    normalised world on configs 3 and 4, where p(w) is the stimulus mixture and an off-centre mu_2
    tips 60.7% of the mass to one side of w = 0, giving H(O) = 0.66997. The log-2 deviation is
    reported beside the identity because it is a fact about those layouts rather than an error.
    """
    t2 = curves[(curves["family"] == "top2") & curves["finite"]]
    chain = (t2["I_AO"] + t2["I_WAgO"] - t2["I_WA"]).abs().max()
    noise = t2["I_AOgW"].abs().max()
    worst_h, off_log2 = 0.0, {}
    for (world, c), g in t2.groupby(["world", "stimulus_config"]):
        world_d, _U, pw = gworld.build(int(c), **world_kws[world])
        po = candidates.top2_pathway(world_d, 0.5)[0] @ pw
        h_o = float(-(po * np.log(po)).sum())
        worst_h = max(worst_h, float((g["I_WO"] - h_o).abs().max()))
        if abs(h_o - np.log(2.0)) > tol:
            off_log2[f"{world}/cfg{int(c)}"] = {"p(o)": [round(float(v), 5) for v in po],
                                                "H(O)": round(h_o, 6)}
    return {"chain |I(W;A) - I(A;O) - I(W;A|O)|": float(chain),
            "router noise max |I(A;O|W)|": float(noise),
            "max |I(W;O) - H(p(o))|": float(worst_h),
            "cut does not halve p(w)": off_log2,
            "ok": bool(chain < tol and noise < tol and worst_h < tol)}


def check_lean(configs, world_kw, betas_head, q, beta3s=(0.3, 1.0, 5.0), tol=1e-12):
    """`score_pathway` must equal `transfer.score` wherever `transfer.score` still works.

    The two differ by one `np.where` on cells the router gives zero weight, so they must agree
    exactly on every defined quantity. This is what stops the masked path from becoming a second,
    silently different accounting -- which is the failure this package's README warns about in its
    own terms ("use true_terms, not policies.info_terms").
    """
    beta1, beta2 = betas_head
    worst, n = 0.0, 0
    for c in configs:
        world_d, _U, _pw = gworld.build(c, **world_kw)
        pogw, pago = candidates.top2_pathway(world_d, q)
        for b3 in beta3s:
            betas = (beta1, beta2, float(b3))
            try:
                want = transfer.score(pogw, pago, world_d["w_vals"], c, betas, world_kw=world_kw)
            except AssertionError:
                continue
            got = score_pathway(pogw, pago, world_d["w_vals"], c, betas, world_kw)
            worst = max(worst, max(abs(got[k] - want[k]) for k in _TERM_KEYS))
            n += 1
    return {"max |lean - transfer.score|": worst, "cells compared": n,
            "ok": bool(n > 0 and worst < tol)}


def check_mirror(curves, tol=1e-6):
    """Configs 3 and 4 are each other's w-mirror and every quantity here is mirror-invariant.

    So their two curves must coincide. A visible gap is not a bug in this module -- it means the
    solve landed on structurally different fixed points on the two layouts, which is a result to
    report rather than to hide, so this returns a number and never raises.
    """
    out = {}
    for world, gw in curves.groupby("world"):
        for fam, gf in gw.groupby("family"):
            a = gf[gf["stimulus_config"] == 3].set_index("beta3").sort_index()
            b = gf[gf["stimulus_config"] == 4].set_index("beta3").sort_index()
            if not len(a) or not len(b) or not a.index.equals(b.index):
                continue
            d = max(float((a[k] - b[k]).abs().max()) for k in ("E_U", "J", "I_WAgO", "I_WA"))
            out[f"{world}/{fam}"] = d
    return {"max_abs_diff_cfg3_vs_cfg4": out,
            "ok": bool(all(v < tol for v in out.values())) if out else None}


def check_top2_is_bayes(configs, world_kw, tol=1e-9):
    """At beta3 = 1 the pinned family's p(a|w) IS the hard top-2 Bayes posterior.

    `test_agreement.check_top2_is_the_posterior` owns this identity; it is re-run here because
    beta3 = 1 is the marked point on every top-2 curve in the figures, and a marker is only worth
    drawing if the thing it marks is exact at THIS module's settings, not at that one's.
    """
    from bounded_optimal_confidence.bounded_rational import seeds as policies
    worst, skipped = 0.0, 0
    for c in configs:
        world_d, U_pre, pw = gworld.build(c, **world_kw)
        pogw, pago = candidates.top2_pathway(world_d, 0.5)
        pagw = marginalizeo_fast(pogw, policies.state_from(pogw, pago, BETA2, 1.0, U_pre, pw)[4])
        post = np.exp(U_pre - U_pre.max(axis=0, keepdims=True))
        post = post / post.sum(axis=0, keepdims=True)
        order = np.sort(post, axis=0)
        # An exactly tied argmin -- the two outer actions at w = 0 on a symmetric grid -- has no
        # fact of the matter about which one the rule discards, and the hard cut has to pick a
        # side. Those columns are excluded and counted, not silently averaged away.
        tied = (order[1] - order[0]) < 1e-12
        trunc = np.where(post == order[0][None, :], 0.0, post)
        trunc = trunc / np.where(trunc.sum(axis=0, keepdims=True) > 0, trunc.sum(axis=0, keepdims=True), 1.0)
        worst = max(worst, float(np.abs(pagw - trunc)[:, ~tied].max()))
        skipped += int(tied.sum())
    return {"max |p(a|w) - hard top-2 posterior| at beta3 = 1": worst,
            "tied-argmin columns skipped": skipped, "ok": bool(worst < tol)}


def run_checks(curves, configs, worlds, betas_head, q):
    """Every check, against a curves table. Separated from `main` so a finished run can be
    re-checked without re-solving it -- `--recheck <dir>`. A check that turns out to be wrong is
    then cheap to fix, which is what happened to the log-2 half of `check_hard_router`."""
    world_kws = {w: world_kw_for(w) for w in worlds}
    return {"hard_router": check_hard_router(curves, world_kws),
            "mirror_cfg3_cfg4": check_mirror(curves),
            "top2_is_bayes_at_beta3_1": check_top2_is_bayes(configs, world_kws[worlds[0]]),
            "lean_vs_transfer_score": check_lean(configs, world_kws[worlds[0]], betas_head, q)}


def _b(x):
    return "inf" if np.isinf(x) else f"{x:g}"


def write_config_log(path, axis, worlds, configs, q, betas_head, want_rd, n_workers, started):
    """config_log.txt, with an authoritative [axis] block FIRST.

    The `settings` dump below the block is the package's constants, which a CLI override does not
    rewrite, so a reader who trusts it over the block gets the default grid rather than this run's.
    """
    beta1, beta2 = betas_head
    L = [f"rate-utility beta3 sweep {started.isoformat(timespec='seconds')}",
         f"git commit: {git_commit()}", "",
         "[axis]  -- authoritative; the config dumps below do NOT describe this run",
         "swept      = beta3",
         f"fixed      = beta1 = {_b(beta1)}, beta2 = {_b(beta2)}, "
         f"card_model = {settings.CARD_MODEL}, q = {q:g}",
         f"beta3 grid = {len(axis)} rungs over [{axis[0]:g}, {axis[-1]:g}] (log), "
         f"marks {list(BETA3_MARKS)} folded in exactly",
         f"worlds     = {list(worlds)}",
         f"configs    = {list(configs)}",
         f"families   = top2, solved (card 2, collapsed branch EXCLUDED)"
         + (", rd frontier" if want_rd else ""),
         f"cells      = {len(axis) * len(worlds) * len(configs)}",
         f"seeds      = {candidates.SEED_MODE} (seeds.init_states), no collapsed branch",
         f"solver     = eps_conv {settings.EPS_CONV:g}, maxiter {settings.FULL_MAXITER}",
         f"workers    = {n_workers}", ""]
    for w in worlds:
        for c in sorted(configs):
            world_d, _U, _pw = gworld.build(c, **world_kw_for(w))
            L += [f"[{w}] " + line for line in gworld.describe(world_d)]
    L += ["", "[settings]"] + settings.config_lines()
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def world_kw_for(world):
    """The world arguments for one world name. Passed per TASK, never through the environment.

    worker_pool spawns, so a module global set in the parent never reaches a child, and this run
    wants two worlds in one pool. Every argument `gworld.build` needs is explicit, so carrying them
    in the task tuple is both sufficient and the only way to interleave the worlds in a single pool.
    """
    kw = {"world": world, "card_model": settings.CARD_MODEL}
    if world == "raw":
        kw.update({"epsilon": settings.EPSILON, "n_samples": settings.N_SAMPLES,
                   "pw_prior": settings.PW_PRIOR})
    kw["sigma"] = settings.SIGMA
    return kw


def append_summary(path, curves, duration, checks):
    solved = curves[curves["family"] == "solved"]
    L = ["", "[run summary]",
         f"cells = {len(curves)} rows, duration = {duration:.1f}s",
         f"family census        = {dict(Counter(curves['family']))}",
         f"non-finite rows      = {int((~curves['finite']).sum())} of {len(curves)}",
         f"stop_reason census   = {dict(Counter(solved['stop_reason'].dropna()))}",
         f"cardinality census   = {dict(Counter(solved['effective_cardinality'].dropna()))}",
         f"winning seed census  = {dict(Counter(solved['seed'].dropna()))}",
         f"chirality census     = {dict(Counter(solved['chirality'].dropna()))}",
         f"self-transfer resid  = max {solved['self_transfer_resid'].max():.3g}, "
         f"median {solved['self_transfer_resid'].median():.3g}",
         f"collapsed would win  = {int((solved['J_collapsed'] > solved['J']).sum())} of {len(solved)}"]
    bad = curves[~curves["finite"]]
    if len(bad):
        L.append(f"WARNING: non-finite from beta3 = {bad['beta3'].min():g} upward "
                 f"(families {sorted(set(bad['family']))})")
    loud = solved[solved["self_transfer_resid"] > SELF_TRANSFER_LOUD]
    if len(loud):
        L.append(f"WARNING: {len(loud)} solved rows re-score more than {SELF_TRANSFER_LOUD:g} "
                 f"from their solve J")
    L += ["", "[checks]"] + [f"{k} = {v!r}" for k, v in checks.items()]
    with open(path, "a") as f:
        f.write("\n".join(L) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Rate-utility curves in beta3: pinned top-2 vs the solved hierarchy.")
    p.add_argument("--worlds", nargs="+", choices=settings.WORLDS, default=list(settings.WORLDS))
    p.add_argument("--configs", type=int, nargs="+", default=None)
    p.add_argument("--beta3-min", type=float, default=None)
    p.add_argument("--beta3-max", type=float, default=None)
    p.add_argument("--beta3-n", type=int, default=None)
    p.add_argument("--q", type=float, default=0.5, help="the top-2 mixing weight; 0.5 is top-2")
    p.add_argument("--beta1", default="inf",
                   help="inf (default) or a finite value, e.g. 1000, if inf will not settle")
    p.add_argument("--no-rd-frontier", action="store_true")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--tag", default="")
    p.add_argument("--recheck", default=None,
                   help="re-run the checks against a finished run's curves.csv and exit")
    args = p.parse_args(argv)

    if args.recheck:
        curves = pd.read_csv(os.path.join(args.recheck, CURVES_NAME))
        worlds = tuple(dict.fromkeys(curves["world"]))
        configs = tuple(sorted(curves["stimulus_config"].unique()))
        head = (float(curves["beta1"].iloc[0]), float(curves["beta2"].iloc[0]))
        checks = run_checks(curves, configs, worlds, head, float(curves["q"].iloc[0]))
        with open(os.path.join(args.recheck, LOG_NAME), "a") as f:
            f.write("\n".join(["", "[checks re-run "
                               + datetime.now().isoformat(timespec="seconds") + "]"]
                              + [f"{k} = {v!r}" for k, v in checks.items()]) + "\n")
        for name, res in checks.items():
            print(f"  check {name}: {res}")
        return args.recheck

    worlds = tuple(dict.fromkeys(args.worlds))
    configs = tuple(args.configs) if args.configs else settings.CONFIGS
    axis = beta3_axis(args.beta3_n, args.beta3_min, args.beta3_max)
    beta1 = float(args.beta1)
    betas_head = (beta1, BETA2)
    want_rd = not args.no_rd_frontier
    n_workers = args.workers or settings.N_WORKERS

    started = datetime.now()
    run_dir = os.path.join(OUTPUT_DIR, f"{RUN_PREFIX}_" + started.strftime("%Y%m%d_%H%M%S")
                           + (f"_{args.tag}" if args.tag else ""))
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, LOG_NAME)
    write_config_log(log_path, axis, worlds, configs, args.q, betas_head, want_rd, n_workers,
                     started)
    print(f"run dir: {run_dir}")
    print(f"{len(axis)} beta3 rungs x {len(configs)} configs x {len(worlds)} worlds = "
          f"{len(axis) * len(configs) * len(worlds)} cells; "
          f"beta1 = {_b(beta1)}, beta2 = {_b(BETA2)}, q = {args.q:g}")

    tasks = [(world_kw_for(w), c, float(b3), args.q, betas_head, want_rd)
             for w in worlds for c in configs for b3 in axis]
    rows = []
    with worker_pool(n_workers) as ex:
        futures = [ex.submit(cell, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(tasks), desc="cells"):
            rows += fut.result()

    curves = pd.DataFrame(rows).sort_values(
        ["world", "family", "stimulus_config", "beta3"]).reset_index(drop=True)
    curves.to_csv(os.path.join(run_dir, CURVES_NAME), index=False)

    checks = run_checks(curves, configs, worlds, betas_head, args.q)
    duration = (datetime.now() - started).total_seconds()
    append_summary(log_path, curves, duration, checks)
    with open(os.path.join(OUTPUT_DIR, LATEST_NAME), "w") as f:
        f.write(run_dir + "\n")

    for name, res in checks.items():
        print(f"  check {name}: {res}")
    print(f"done in {duration:.1f}s -> {run_dir}")
    return run_dir
