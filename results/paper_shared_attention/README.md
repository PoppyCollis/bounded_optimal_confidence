# The paper's shared-attention B sweep

`optimise_attention_discrete_Bsweep_recentred_n10000_sig64_20261025_150919/` is the sweep the p(x|phi) figure is
drawn from (`paper_plots/fig_pxgphi_vs_B.py`): `sweep.csv`, made by this package on 2026-10-25 on 2 workers. It is
the re-run of the paper's original sweep (`ITBR_confidence/generalisation/results/`, 2026-10-12, in git history to
commit 75a4496) and matches it in every column but the two timing ones. The folder name is kept: the figure reads
the prior and the `_discrete` tag from it. The same sweep is

    python scripts/optimise_attention_discrete.py --world-prior recentred --sweep-B 64 125 250 500 1000 2000 --n-seeds 5 --n-workers 2

## The model, in short

Worlds c have three category means (-a, m, a), drawn by the uniform recentred prior: three i.i.d. U(-B, B) draws,
sorted, shifted so the outer pair is at ±a. The observation w is discrete on a grid over [-B - 5σ, B + 5σ] and
p_c(w|x) is the Gaussian N(w; μ_cx, σ²), σ = 64, normalised over the grid. Two structures φ share a sparse prior
p(x|φ) across worlds (each puts zero on one or two states, so each lives on an edge of the probability triangle),
and a router p(φ|w) shared across worlds picks the structure. J is the expected log-likelihood of the observation
under the attended posterior, E_c Σ_w p_c(w) Σ_φ p(φ|w) E_{p_c(x|w,φ)}[ln p_c(w|x)]. For fixed priors the best
router is hard, so each candidate prior is scored by one A-curve over w and every pair of priors by the sum over
w of the upper envelope. See `docs/shared_attention.md`.

## The sweep

| | |
|---|---|
| axis | B = 64, 125, 250, 500, 1000, 2000 (B/σ = 1 .. 31) |
| seeds | 0, 2, 4, 6, 8 for the fit worlds; each held out on seed + 1 |
| worlds | 10,000 per draw; grid step held at 2.85/64 σ (`n_grid` per row) |
| search | 201 points along each edge, best pair for every edge pair, Nelder-Mead refinement; runner-up = best pair with other supports |
| codes scored | `none` (no attention, p(x|φ) = p0), `ref` ([.5,.5,0] / [0,.5,.5] with the switch at 0), `opt` (the sparse optimum), `second` (the runner-up); each as J and as the gap to the Dirac-delta bound J_δ − J ≥ 0 |

Held-out means over seeds (from the original notes; `summarise` prints the same table at the end of a run):

| B | B/σ | supports | middle weight | gap opt | gap ref | gap none |
|---|---|---|---|---|---|---|
| 64 | 1.0 | {left} / {right} | 0 | 0.0111 | 0.0995 | 0.2031 |
| 125 | 2.0 | {left} / {right} | 0 | 0.0566 | 0.1253 | 0.2205 |
| 250 | 3.9 | {L,M} / {M,R} | 0.320 | 0.1157 | 0.1275 | 0.1743 |
| 500 | 7.8 | {L,M} / {M,R} | 0.433 | 0.0927 | 0.0942 | 0.1091 |
| 1000 | 15.6 | {L,M} / {M,R} | 0.470 | 0.0567 | 0.0569 | 0.0610 |
| 2000 | 31.3 | {L,M} / {M,R} | 0.486 | 0.0314 | 0.0314 | 0.0325 |

The router switches at w = 0 in every row (stored to one grid step; one row, B 125 seed 6, stores the cut one step
along: a tie at 0). The continuous-w version of the script (`optimise_attention.py`, not carried over) gave the
same sweep: same supports and grid in all 30 rows, weights within 5e-7, gaps within 2e-8, J shifted by ln(grid step).

## Columns of `sweep.csv`

| column | meaning |
|---|---|
| `B`, `seed`, `n_grid` | the job: box size, seed of the fit worlds (held-out worlds use seed + 1), grid points |
| `pi1_left`, `pi1_middle`, `pi1_right`, `pi2_*` | the two optimal structure priors p(x|φ); φ₁ is the one further left |
| `lm_mr` | 1 when φ₁ excludes right and φ₂ excludes left (so each has a middle weight) |
| `supports`, `second_supports` | the supports of the optimum and of the runner-up |
| `switches` | the w grid points at which the hard router changes structure (the first point routed to the other one) |
| `J_{delta,none,ref,opt,second}_{fit,held}` | J of each code on the fit worlds and the held-out worlds; `delta` is the Dirac-delta bound |
| `gap_{none,ref,opt,second}_{fit,held}` | `J_delta − J` of each code |
| `seconds`, `peak_gb` | the job's wall time and peak memory (not reproducible) |

`python scripts/check_paper_shared_attention.py` re-solves the jobs with this package and compares every other
column with the stored rows exactly; `pytest -m slow -k shared_attention` re-solves one.
