# The shared attention optimum over sampled worlds

Code: `src/bounded_optimal_confidence/shared_attention/optimise_attention_discrete.py` (a self-contained script:
numpy, scipy, matplotlib). Scripts: `scripts/optimise_attention_discrete.py` (one B, or the B sweep),
`scripts/check_paper_shared_attention.py` (reproduce the paper's sweep), `paper_plots/fig_pxgphi_vs_B.py` (the
figure). The paper's sweep is `results/paper_shared_attention/` (its README has the table and the columns).

## The question

If one attention mechanism has to serve many category layouts, which is best? The layouts are worlds c with means
(−a, m, a) drawn from the uniform recentred prior (three i.i.d. U(−B, B) draws, sorted, shifted so the outer pair is
at ±a; B is the only free scale, and only B/σ matters). A mechanism is two *structures* φ, each a prior p(x|φ) over
the three states, shared across worlds, plus a router p(φ|w) shared across worlds. The structures are **sparse**:
each gives zero probability to one or two states, so each lives on an edge of the probability triangle, parametrised
by the edge e and a position t ∈ [0, 1] along it (t = 0 or 1 is a one-state prior).

## The objective

The observation w is discrete, taking the values of a grid W over [−B − 5σ, B + 5σ] whose step is held at 2.85/64 σ
in the sweep. p_c(w|x) = N(w; μ_cx, σ²) normalised over W, p_c(w) = Σ_x p0(x) p_c(w|x) with p0 uniform, and the
attended posterior is p_c(x|w, φ) ∝ p(x|φ) p_c(w|x). The score is the expected log-likelihood of the observation
under that posterior,

    J = E_c Σ_w p_c(w) Σ_φ p(φ|w) E_{p_c(x|w,φ)}[ln p_c(w|x)].

For fixed (π₁, π₂) the best router is hard: p(φ|w) = 1[φ = argmax_φ A(w, φ)] with A(w, φ) = E_c[p_c(w) F_c(w, φ)]
and F the inner expectation. A(·, φ) depends on π_φ alone, so one **A-curve** is computed per candidate prior and
every pair is scored by Σ_w max_φ A(w, φ). (`A_curve` works on likelihoods relative to the best state at each (c, w)
and redoes underflowed cells in log space, which narrow σ needs.)

Discrete w rather than the integral: the continuous version of the script (left behind) agrees with this one to
5e-7 in the weights and 2e-8 in the gaps, with J shifted by exactly ln(grid step).

## The search (`solve`)

For each of the six edge pairs (φ labels interchangeable), the A-curves are tabulated at 201 positions along each
edge, the best (t₁, t₂) on that grid is refined by Nelder-Mead (kept only if it improves), and the pairs are ranked
by J. The optimum is the best pair, with φ₁ the structure further left; the **runner-up** is the best pair whose
supports differ from the optimum's. Four codes are scored on the fit worlds and on an independent held-out draw
(seed + 1): `none` (p(x|φ) = p0, no attention), `ref` (the code [.5, .5, 0] / [0, .5, .5] with the switch at 0),
`opt` and `second`. Each is reported as J and as its gap to the **Dirac-delta bound** J_δ − J ≥ 0, where J_δ puts all
belief on the best-explaining state at each (c, w). The gap, not the gain over no attention, is how codes are judged.

## The B sweep (`run_sweep`)

One job per (B, seed), seeds 0, 2, 4, 6, 8, worlds 10,000, in a process pool with one task per process (a B = 2000
job holds 2.3 GB). `sweep.csv` has one row per job; `summarise` prints the per-B means over seeds on the held-out
worlds (the middle weights, their distance from 0.5 and its local log-log slope, the gaps, the share of the
no-attention gap each code closes) and `plot_sweep` draws them (`B_sweep.png`).

What the paper's sweep found (recentred prior, σ = 64): the supports change from {left} / {right} at B/σ ≤ 2 to
{left, middle} / {middle, right} from B/σ ≈ 4 on, the middle weight rises towards 0.5 from below (0.32, 0.43, 0.47,
0.486 at B/σ = 3.9, 7.8, 15.6, 31), the router switches at w = 0 throughout, and the reference code
[.5, .5, 0] / [0, .5, .5] is within 0.0015 nats of the optimum from B/σ ≈ 8.

## Running it

    python scripts/optimise_attention_discrete.py --world-prior recentred --B 250              # one B, ~1 min: attention_optimum.png, pogw_pago.png
    python scripts/optimise_attention_discrete.py --world-prior recentred \
        --sweep-B 64 125 250 500 1000 2000 --n-seeds 5 --n-workers 2                           # the paper's sweep, ~40 min
    python scripts/check_paper_shared_attention.py [--jobs 64:0 250:0] [--workers 2]           # reproduce stored rows exactly
    python paper_plots/fig_pxgphi_vs_B.py [--run DIR | --latest]                               # the figure
    pytest tests/test_shared_attention.py; pytest -m slow -k shared_attention

The sweep is deterministic: `np.random.default_rng(seed)` draws the worlds, and nothing else is random. The check
script compares every column of `sweep.csv` but the two timing ones with the stored rows and reproduces all 30.

## The figure (`paper_plots/fig_pxgphi_vs_B.py`)

Three panels against B/σ on a log-2 axis: p(x|φ₁) and p(x|φ₂), one line per state (mean over seeds, faint dots the
seeds, dashed line at 0.5), and the switch-point of the router p(φ|y) in units of σ (the stimulus w is labelled y in
the figure). The cut is known to one grid step: `sweep.csv` stores the first grid point routed to the other
structure, and the figure plots the middle of the cell before it. Written as
`paper_plots/figures/pxgphi_vs_B_recentred_discrete.{png,svg}`.
