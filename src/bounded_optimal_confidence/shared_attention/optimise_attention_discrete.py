"""
Option 3 with sparse structure priors: profile out the router, then search
over the two structure priors.

Model
  worlds   means (-a, m, a), from one of three priors (WORLD_PRIOR), all sampled in this file:
             "recentred": draw 3 means i.i.d. U(-B, B), sort, shift by the midrange
             "m_first":   m ~ U(-B, B), then a ~ U(|m|, B)
             "a_first":   a ~ U(0, B), then m ~ U(-a, a)
  obs      w is discrete: it takes the N_GRID values of a uniform grid W over [-B - 5s, B + 5s],
             with s = SIGMA, or the upper end of SIGMA_RANGE
  lik      p_c(w|x) = N(w; mu_{c,x}, sigma_{c,x}^2) / sum_{w' in W} N(w'; mu_{c,x}, sigma_{c,x}^2),
             a Gaussian discretised onto W, so p_c(w|x) is a probability over W
             sigma fixed (SIGMA), or with SIGMA_RANGE = (lo, hi) each world draws its three
             sigmas i.i.d. U(lo, hi), one per state
  data     p_c(w) = sum_x p0(x) p_c(w|x),  p0 uniform
  attn     shared priors pi_phi = p(x|phi), phi in {1,2}; shared router p(phi|w)
  J        = E_c sum_w p_c(w) sum_phi p(phi|w) E_{p_c(x|w,phi)}[ln p_c(w|x)]

Sparsity: each pi_phi must give zero probability to 1 or 2 states. The feasible
set for each structure is therefore the three edges of the probability
triangle (endpoints = one-state priors). Each structure is parametrised by an
edge e and a position t in [0, 1] along it.

For fixed (pi_1, pi_2) the optimal router is hard:
  p(phi|w) = 1[phi = argmax A(w,phi)],   A(w,phi) = E_c[p_c(w) F_c(w,phi)]
so J*(pi_1, pi_2) = sum_w max_phi A(w,phi).
A(., phi) depends only on pi_phi, so we compute one A-curve per candidate
prior and score every pair by the sum over w of the upper envelope.

    python scripts/optimise_attention_discrete.py --world-prior m_first --B 250 --sigma 64
    python scripts/optimise_attention_discrete.py --sigma-range 32 96     # random sigmas per world
    python scripts/optimise_attention_discrete.py --sweep-B 64 125 250 500 1000 2000 --n-seeds 5 \
        --n-workers 2

Outputs (results/shared_attention/optimise_attention_discrete_<prior>_n<n>_s<seed>_B<B>_sig<sigma | lo-hi>_<time>/):
  attention_optimum.png   A-curves with the optimal router, and the two structure priors
  pogw_pago.png           one panel: the router p(phi|w) (labelled A) beside p(x|phi), one colour bar

B sweep (--sweep-B): the same search at each B for seeds 0, 2, 4, ... (held-out worlds use seed + 1),
with the w grid step held at SWEEP_STEP sigmas. Every code is scored by its gap to the Dirac-delta
bound, J_delta - J >= 0, where J_delta puts all belief on the best-explaining state at each (c, w).
Outputs (results/shared_attention/optimise_attention_discrete_Bsweep_<prior>_n<n>_sig<sigma | lo-hi>_<time>/):
  sweep.csv     one row per (B, seed): the optimum, J and the gap of every code, fit and held-out

From ITBR_confidence/generalisation/optimise_attention_discrete.py, unchanged but for the output directory and these
run lines. The paper's sweep is results/paper_shared_attention/optimise_attention_discrete_Bsweep_recentred_n10000_sig64_20261025_150919
(recentred prior, sigma 64, 10000 worlds, seeds 0 2 4 6 8, B 64 .. 2000); paper_plots/fig_pxgphi_vs_B.py draws from it.
"""
import argparse
import csv
import itertools
import multiprocessing
import os
import resource
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

# ---------------- config (defaults; all but P0 and REFERENCE have a CLI flag) ----------------
B = 250.0         # bound on the means (both priors)
SIGMA = 64.0      # likelihood std (fixed across worlds and states)
SIGMA_RANGE = None  # (lo, hi): draw each world's three sigmas i.i.d. U(lo, hi) instead of SIGMA
N_WORLDS = 10000  # Monte Carlo samples c ~ p(c)
N_GRID = 401      # number of discrete observation values w
N_T = 201         # grid points along each edge for the global search
SEED = 0
WORLD_PRIOR = "m_first"  # "recentred", "m_first" or "a_first"
P0 = np.full(3, 1 / 3)  # generative state prior p0(x)
REFERENCE = (np.array([.5, .5, 0]), np.array([0, .5, .5]), 0.0)  # (pi1, pi2, switch w)

STATES = ["left", "middle", "right"]
EDGES = [(0, 1), (0, 2), (1, 2)]  # supports of size 2; t=0 or 1 gives support 1
EDGE_NAMES = ["{left, middle}", "{left, right}", "{middle, right}"]

from bounded_optimal_confidence.paths import SHARED_ATTENTION_DIR

OUTPUT_DIR = str(SHARED_ATTENTION_DIR)

SWEEP_STEP = 2.85 / 64  # B sweep: w grid step in units of sigma (the default grid's step)
CODES = {"none": "no attention", "ref": "reference (pi1, pi2, switch)", "opt": "sparse optimum",
         "second": "runner-up supports"}
# Chart ink and categorical slots 1-5 of the dataviz reference palette, fixed by entity.
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3dd", "#fcfcfb"
COLOURS = {"phi1": "#2a78d6", "phi2": "#eb6834", "opt": "#1baf7a", "ref": "#eda100",
           "second": "#e87ba4", "none": MUTED}


def make_grid():
    """The discrete observation values w and their spacing; the outer means lie in [-B, B]."""
    half = B + 5 * (SIGMA_RANGE[1] if SIGMA_RANGE else SIGMA)
    w = np.linspace(-half, half, N_GRID)
    return w, w[1] - w[0]


W, DW = make_grid()


def sample_recentred(n, bound, rng):
    """Uniform recentred prior: sort 3 i.i.d. U(-B, B) means, shift outer pair to ±a."""
    raw = np.sort(rng.uniform(-bound, bound, size=(n, 3)), axis=1)
    midrange = 0.5 * (raw[:, 0] + raw[:, 2])
    return raw - midrange[:, None]


def sample_m_first(n, bound, rng):
    """m-first prior: m ~ U(-B, B), then a ~ U(|m|, B)."""
    m = rng.uniform(-bound, bound, size=n)
    a = rng.uniform(np.abs(m), bound)
    return np.column_stack([-a, m, a])


def sample_a_first(n, bound, rng):
    """a-first prior: a ~ U(0, B), then m ~ U(-a, a)."""
    a = rng.uniform(0, bound, size=n)
    m = rng.uniform(-a, a)
    return np.column_stack([-a, m, a])


SAMPLERS = {"recentred": sample_recentred, "m_first": sample_m_first, "a_first": sample_a_first}


def sample_sigmas(n, rng):
    """Likelihood std per world and state: SIGMA, or 3 i.i.d. U(lo, hi) draws per world."""
    if SIGMA_RANGE is None:
        return np.full((n, 3), SIGMA)
    return rng.uniform(*SIGMA_RANGE, size=(n, 3))


def build_worlds(seed):
    """Sample worlds and precompute over the discrete w: the likelihoods relative to the best state
    at each (c, w), those times ln p_c(w|x), ln p_c(w|x) itself, and p_c(w)."""
    rng = np.random.default_rng(seed)
    mu = SAMPLERS[WORLD_PRIOR](N_WORLDS, B, rng)                           # (N,3)
    sigma = sample_sigmas(N_WORLDS, rng)                                   # (N,3), after the means
    log_gauss = -0.5 * ((W[None, :, None] - mu[:, None, :]) / sigma[:, None, :]) ** 2  # unnormalised
    loglik = log_gauss - logsumexp(log_gauss, axis=1, keepdims=True)       # (N,K,3), normalised over w
    del log_gauss
    pw = np.exp(loglik) @ P0                                               # (N,K), sums to 1 over w
    rel = np.exp(loglik - loglik.max(2, keepdims=True))  # the scale cancels in F
    return rel, rel * loglik, loglik, pw


def A_curve(pi, worlds):
    """A(w) = E_c[p_c(w) F_c(w)] for one structure prior pi on the 3 states."""
    rel, rel_loglik, loglik, pw = worlds
    den = rel @ pi
    with np.errstate(divide="ignore", invalid="ignore"):
        F = (rel_loglik @ pi) / den    # E_{p_c(x|w,phi)}[ln p_c(w|x)]
    bad = den < 1e-200  # every state pi allows is far below the best one: redo these cells in logs
    if bad.any():
        sup = pi > 0
        ll = loglik[bad][:, sup]
        z = ll + np.log(pi[sup])
        r = np.exp(z - z.max(1, keepdims=True))
        F[bad] = (r * ll).sum(1) / r.sum(1)
    return (pw * F).mean(0)            # (K,)


def J_star(pi1, pi2, worlds):
    return np.maximum(A_curve(pi1, worlds), A_curve(pi2, worlds)).sum()


def edge_prior(e, t):
    """Prior on edge e at position t: mass 1-t on the first state, t on the second."""
    pi = np.zeros(3)
    i, j = EDGES[e]
    pi[i], pi[j] = 1 - t, t
    return pi


def configure(**kw):
    """Set config globals (keys are the constants' names) and rebuild the w grid."""
    global W, DW
    globals().update(kw)
    W, DW = make_grid()


def support_name(pi):
    return "{" + ", ".join(n for n, v in zip(STATES, pi) if v > 0) + "}"


def solve():
    """Search the sparse priors under the current config. Returns the ranked support pairs, the
    optimum with its A-curves and router, and J of every code on the fit and held-out worlds."""
    worlds = build_worlds(SEED)
    T = np.linspace(0, 1, N_T)
    curves = [np.stack([A_curve(edge_prior(e, t), worlds) for t in T]) for e in range(3)]

    # ---- best pair for every combination of edges (phi labels interchangeable) ----
    results = []
    for e1, e2 in itertools.combinations_with_replacement(range(3), 2):
        c1, c2 = curves[e1], curves[e2]
        scores = np.stack([np.maximum(c1[k], c2).sum(1) for k in range(N_T)])
        k1, k2 = np.unravel_index(np.argmax(scores), scores.shape)

        def neg_J(t, e1=e1, e2=e2):
            return -J_star(edge_prior(e1, t[0]), edge_prior(e2, t[1]), worlds)

        res = minimize(neg_J, [T[k1], T[k2]], method="Nelder-Mead",
                       bounds=[(0, 1), (0, 1)],
                       options=dict(xatol=1e-6, fatol=1e-10, maxiter=2000))
        if -res.fun < scores[k1, k2]:  # keep the grid point if refinement didn't help
            res.x, res.fun = np.array([T[k1], T[k2]]), -scores[k1, k2]
        results.append((-res.fun, edge_prior(e1, res.x[0]), edge_prior(e2, res.x[1]), e1, e2))
    results.sort(key=lambda r: -r[0])

    # ---- the overall optimum; phi_1 is the structure further left ----
    pi1, pi2 = results[0][1:3]
    if pi1 @ [-1, 0, 1] > pi2 @ [-1, 0, 1]:
        pi1, pi2 = pi2, pi1
    a1, a2 = A_curve(pi1, worlds), A_curve(pi2, worlds)
    route1 = a1 > a2                                       # p(phi=1|w) = 1 here
    switches = W[1:][np.diff(route1.astype(int)) != 0]
    def supports(r):
        return {tuple(r[1] > 0), tuple(r[2] > 0)}

    # runner-up: the best pair with other supports (a one-state optimum lies on several edge pairs)
    second = next(r for r in results[1:] if supports(r) != supports(results[0]))
    r1, r2, r_sw = REFERENCE

    def score(wd):
        return dict(delta=(wd[3] * wd[2].max(2)).mean(0).sum(),        # delta on best x per (c,w)
                    none=A_curve(P0, wd).sum(),                         # no attention: p(x|phi)=p0
                    ref=np.where(W < r_sw, A_curve(r1, wd), A_curve(r2, wd)).sum(),
                    opt=J_star(pi1, pi2, wd),
                    second=J_star(second[1], second[2], wd))

    fit = score(worlds)
    del worlds, curves                                     # free the fit worlds before the held-out ones
    return dict(results=results, second=second, pi1=pi1, pi2=pi2, a1=a1, a2=a2, route1=route1,
                switches=switches, fit=fit, held=score(build_worlds(SEED + 1)))


# ---------------- B sweep ----------------
def n_grid_for(bound):
    """Odd grid size whose step is at most SWEEP_STEP of the smallest sigma."""
    sig_lo, sig_hi = SIGMA_RANGE if SIGMA_RANGE else (SIGMA, SIGMA)
    return 2 * int(np.ceil((bound + 5 * sig_hi) / (SWEEP_STEP * sig_lo) - 1e-9)) + 1


def _sweep_job(job):
    cfg, bound, seed = job
    configure(**cfg, B=bound, SEED=seed)
    configure(N_GRID=n_grid_for(bound))
    t0 = time.time()
    r = solve()
    pi1, pi2 = r["pi1"], r["pi2"]
    row = dict(B=bound, seed=seed, n_grid=N_GRID)
    row.update({f"pi{i}_{n}": v for i, pi in ((1, pi1), (2, pi2)) for n, v in zip(STATES, pi)})
    # lm_mr: phi_1 excludes right and phi_2 excludes left, so each has a middle weight (0 if one-state)
    row.update(lm_mr=int(pi1[2] == 0 and pi2[0] == 0),
               supports=f"{support_name(pi1)} | {support_name(pi2)}",
               second_supports=f"{support_name(r['second'][1])} | {support_name(r['second'][2])}",
               switches="; ".join(f"{w:.6g}" for w in r["switches"]))
    for part in ("fit", "held"):
        row.update({f"J_{k}_{part}": v for k, v in r[part].items()})
        row.update({f"gap_{k}_{part}": r[part]["delta"] - r[part][k] for k in CODES})
    row.update(seconds=time.time() - t0,
               peak_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20)
    return row


def summarise(rows):
    """Per-B means over seeds, held-out worlds. The weight columns use only the seeds whose optimum
    has phi_1 without right and phi_2 without left; dev is 0.5 minus the mean of the two middle weights."""
    scale = 1.0 if SIGMA_RANGE else SIGMA
    out = []
    for b in sorted({r["B"] for r in rows}):
        g = [r for r in rows if r["B"] == b]
        mid = np.array([[r["pi1_middle"], r["pi2_middle"]] for r in g if r["lm_mr"]]).reshape(-1, 2)
        n = len(mid)
        dev = 0.5 - mid.mean(1)
        gap = {k: np.array([r[f"gap_{k}_held"] for r in g]) for k in CODES}
        closed = {k: 1 - gap[k] / gap["none"] for k in ("opt", "ref")}
        out.append(dict(B=b, x=b / scale, n=len(g), n_lm_mr=n,
                        mid1=mid[:, 0].mean() if n else np.nan, mid2=mid[:, 1].mean() if n else np.nan,
                        dev=dev.mean() if n else np.nan,
                        dev_se=dev.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan,
                        gap={k: v.mean() for k, v in gap.items()},
                        closed={k: v.mean() for k, v in closed.items()},
                        closed_sd={k: v.std(ddof=1) if len(g) > 1 else np.nan for k, v in closed.items()}))
    return out


def local_slopes(summary):
    """Slope of log(0.5 - middle weight) on log(B) between consecutive B values (nan where dev <= 0)."""
    x, dev = np.array([[s["x"], s["dev"]] for s in summary]).T
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.diff(np.log(dev)) / np.diff(np.log(x))


def run_sweep(args):
    cfg = dict(SIGMA=args.sigma, SIGMA_RANGE=args.sigma_range, N_WORLDS=args.n_worlds, N_T=args.n_t,
               WORLD_PRIOR=args.world_prior)
    configure(**cfg)
    seeds = [2 * i for i in range(args.n_seeds)]  # held-out worlds use seed + 1
    jobs = [(cfg, b, s) for s in seeds for b in sorted(set(args.sweep_B), reverse=True)]
    print(f"B sweep: {len(jobs)} jobs on {args.n_workers} workers")
    rows = []
    with multiprocessing.Pool(args.n_workers, maxtasksperchild=1) as pool:
        for row in pool.imap_unordered(_sweep_job, jobs):
            rows.append(row)
            print(f"  [{len(rows):3d}/{len(jobs)}] B {row['B']:g} seed {row['seed']}: {row['supports']}, "
                  f"gap {row['gap_opt_held']:.4f}, {row['seconds']:.0f} s, {row['peak_gb']:.2f} GB", flush=True)
    rows.sort(key=lambda r: (r["B"], r["seed"]))

    sig_tag = f"{SIGMA_RANGE[0]:g}-{SIGMA_RANGE[1]:g}" if SIGMA_RANGE else f"{SIGMA:g}"
    run_dir = os.path.join(OUTPUT_DIR, f"optimise_attention_discrete_Bsweep_{WORLD_PRIOR}_n{N_WORLDS}"
                                       f"_sig{sig_tag}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "sweep.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    summary = summarise(rows)
    slopes = [np.nan, *local_slopes(summary)]
    print(f"\nheld-out worlds, mean over seeds; gap = J_delta - J in nats; slope = d ln(0.5-mid) / d ln B "
          f"from the previous B\n{'B':>6s}{'B/sig':>7s}{'LM|MR':>7s}{'mid phi1':>10s}{'mid phi2':>10s}"
          f"{'0.5-mid':>10s}{'s.e.':>9s}{'slope':>8s}{'x B/sig':>9s}"
          f"{'gap none':>10s}{'gap ref':>10s}{'gap opt':>10s}{'gap 2nd':>10s}{'closed opt':>12s}{'closed ref':>12s}")
    for s, sl in zip(summary, slopes):
        print(f"{s['B']:6g}{s['x']:7.2f}{s['n_lm_mr']:4d}/{s['n']:<2d}{s['mid1']:10.4f}{s['mid2']:10.4f}"
              f"{s['dev']:10.4f}{s['dev_se']:9.4f}{sl:8.2f}{s['dev'] * s['x']:9.3f}"
              + "".join(f"{s['gap'][k]:10.4f}" for k in CODES)
              + f"{100 * s['closed']['opt']:11.2f}%{100 * s['closed']['ref']:11.2f}%")
    print(f"\nsaved {run_dir}")


def plot_matrices(route1, pi1, pi2, path):
    """One panel: the router p(phi|w) (structures phi_1, phi_2 from the top, by the w grid; labelled A)
    beside p(x|phi) (states left, middle, right from the top, by structures), sharing one colour bar.
    Laid out like the repo's p(o|w) and p(a|o) matrices (utils.plots) but in warm grey, redrawn here
    so this script stays self-contained."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pogw = np.vstack([route1, ~route1]).astype(float)  # (2 structures, K); the router is hard
    pxgphi = np.column_stack([pi1, pi2])               # (3 states, 2 structures); columns sum to 1
    fs, fs_cell, edge = 30, 20, "#B4B2A9"
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "prob", ["#444441", edge, "#FFFFFF"])  # warm grey, 0 = dark, 1 = white
    lo, hi = W[0] - DW / 2, W[-1] + DW / 2  # the w grid's cell edges
    fig = plt.figure(figsize=(18, 4.2))
    ax_w = fig.add_axes([0.06, 0.3, 0.52, 0.6])
    ax_x = fig.add_axes([0.69, 0.3, 0.17, 0.6])
    for ax, mat, extent in ((ax_w, pogw, (lo, hi, 1.5, -0.5)), (ax_x, pxgphi, None)):
        im = ax.imshow(mat, aspect="auto", cmap=cmap, interpolation="nearest", vmin=0, vmax=1,
                       extent=extent)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(edge)
            spine.set_linewidth(1.5)

    def cell_text(ax, x, y, v):
        ax.text(x, y, f"{v:.2f}", ha="center", va="center", fontsize=fs_cell,
                color="white" if v < 0.25 else "black")

    # p(phi|w): one value per block of constant routing, where the block is wide enough to hold it
    cuts = np.flatnonzero(np.diff(route1.astype(int))) + 1
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(W)]):
        if (b - a) * DW > 0.06 * (hi - lo):
            for row in range(2):
                cell_text(ax_w, 0.5 * (W[a] + W[b - 1]), row, pogw[row, a])
    ax_w.set_xticks([lo, 0, hi])
    ax_w.set_xticklabels(["", "0", ""])
    ax_w.tick_params(axis="x", length=8, width=1.5, color=edge, labelsize=fs, pad=8)
    ax_w.set_xlabel("Data $w$", fontsize=fs, labelpad=10)
    ax_w.set_ylabel(r"Structure $\phi$", fontsize=fs, labelpad=20)

    ax_x.set_xlabel(r"Structure $\phi$", fontsize=fs, labelpad=20)
    ax_x.set_ylabel("State $x$", fontsize=fs, labelpad=20)
    for (i, k), v in np.ndenumerate(pxgphi):
        cell_text(ax_x, k, i, v)

    cbar = fig.colorbar(im, cax=fig.add_axes([0.885, 0.3, 0.012, 0.6]), ticks=[0, 1])
    cbar.set_label("Probability", fontsize=fs, labelpad=10)
    cbar.ax.tick_params(labelsize=fs, length=0)
    cbar.outline.set_edgecolor(edge)
    cbar.outline.set_linewidth(1.5)

    fig.canvas.draw()  # place the panel letter under the left matrix's x label
    box = ax_w.xaxis.label.get_window_extent().transformed(fig.transFigure.inverted())
    fig.text(0.5 * (box.x0 + box.x1), box.y0 - 0.06, "A", ha="center", va="top", fontsize=fs,
             fontweight="bold")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Sparse structure priors p(x|phi) and the optimal shared "
                                            "router over sampled worlds.")
    p.add_argument("--world-prior", choices=sorted(SAMPLERS), default=WORLD_PRIOR, help="world prior")
    p.add_argument("--B", type=float, default=B, help="bound on the means")
    p.add_argument("--sigma", type=float, default=SIGMA, help="likelihood std")
    p.add_argument("--sigma-range", type=float, nargs=2, default=SIGMA_RANGE, metavar=("LO", "HI"),
                   help="draw each world's three sigmas i.i.d. U(LO, HI); overrides --sigma")
    p.add_argument("--n-worlds", type=int, default=N_WORLDS, help="number of sampled worlds")
    p.add_argument("--n-grid", type=int, default=N_GRID, help="w grid points")
    p.add_argument("--n-t", type=int, default=N_T, help="grid points along each edge")
    p.add_argument("--seed", type=int, default=SEED, help="seed of the world draw (held-out: seed + 1)")
    p.add_argument("--sweep-B", type=float, nargs="+", default=None, metavar="B",
                   help="run the B sweep over these bounds; --B, --seed and --n-grid are then set by it")
    p.add_argument("--n-seeds", type=int, default=5, help="B sweep: seeds per B (0, 2, 4, ...)")
    p.add_argument("--n-workers", type=int, default=4,
                   help="B sweep: parallel jobs; memory-bound, each job holds its worlds on the grid")
    args = p.parse_args(argv)
    if args.sigma_range is not None and not 0 < args.sigma_range[0] <= args.sigma_range[1]:
        p.error("--sigma-range needs 0 < LO <= HI")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.sweep_B:
        return run_sweep(args)
    configure(B=args.B, SIGMA=args.sigma, SIGMA_RANGE=args.sigma_range, N_WORLDS=args.n_worlds,
              N_GRID=args.n_grid, N_T=args.n_t, SEED=args.seed, WORLD_PRIOR=args.world_prior)
    sigma_min = SIGMA_RANGE[0] if SIGMA_RANGE else SIGMA
    if DW > sigma_min:
        print(f"warning: w grid step {DW:.3g} exceeds the smallest sigma {sigma_min:g}; raise --n-grid")

    r = solve()
    pi1, pi2, fit, held = r["pi1"], r["pi2"], r["fit"], r["held"]
    np.set_printoptions(precision=3, suppress=True)
    print("best J for each pair of supports")
    for J, p1, p2, e1, e2 in r["results"]:
        print(f"  {J:9.4f}   {EDGE_NAMES[e1]:16s} {p1}   {EDGE_NAMES[e2]:16s} {p2}")

    print("\nsparse optimum")
    print(f"  pi1 = p(x|phi=1) = {pi1}")
    print(f"  pi2 = p(x|phi=2) = {pi2}")
    print(f"  router switches at w = {np.round(r['switches'], 1) + 0.0}")
    print(f"\n{'':32s}{'fit worlds':>12s}{'held-out':>12s}")
    for k in ("none", "ref", "opt"):
        print(f"  {CODES[k]:30s}{fit[k]:12.4f}{held[k]:12.4f}")
    print(f"  {'upper bound (Dirac delta)':30s}{fit['delta']:12.4f}{held['delta']:12.4f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8),
                               gridspec_kw=dict(width_ratios=[2.2, 1]))
        ax[0].plot(W, r["a1"], label=r"$A(w,\phi_1)$")
        ax[0].plot(W, r["a2"], label=r"$A(w,\phi_2)$")
        ax[0].fill_between(W, *ax[0].get_ylim(), where=r["route1"], alpha=0.12,
                           color="C0", label=r"routed to $\phi_1$")
        ax[0].set(xlabel="w", title="Structure scores and optimal router")
        ax[0].legend(fontsize=8)
        x = np.arange(3)
        ax[1].bar(x - 0.2, pi1, 0.4, label=r"$\phi_1$")
        ax[1].bar(x + 0.2, pi2, 0.4, label=r"$\phi_2$")
        ax[1].set(xticks=x, xticklabels=STATES, ylim=(0, 1), title=r"$p(x|\phi)$ (sparse)")
        ax[1].legend(fontsize=8)
        sig_title = f"σ ~ U({SIGMA_RANGE[0]:g}, {SIGMA_RANGE[1]:g})" if SIGMA_RANGE else f"σ = {SIGMA:g}"
        fig.suptitle(f"{WORLD_PRIOR} world prior, B = {B:g}, {sig_title}")
        fig.tight_layout()
        sig_tag = f"{SIGMA_RANGE[0]:g}-{SIGMA_RANGE[1]:g}" if SIGMA_RANGE else f"{SIGMA:g}"
        run_dir = os.path.join(OUTPUT_DIR, f"optimise_attention_discrete_{WORLD_PRIOR}_n{N_WORLDS}_s{SEED}"
                                           f"_B{B:g}_sig{sig_tag}_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(run_dir, exist_ok=True)
        path = os.path.join(run_dir, "attention_optimum.png")
        fig.savefig(path, dpi=150)
        print(f"\nsaved {path}")
        path = os.path.join(run_dir, "pogw_pago.png")
        plot_matrices(r["route1"], pi1, pi2, path)
        print(f"saved {path}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
