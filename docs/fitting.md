# The fit

Code: `src/bounded_optimal_confidence/fitting/` and `likelihood/`. Run it with `scripts/fit_model.py`.

## What is fitted

Each subject is fitted on all four configurations of experiment 1 together, with **one parameter set**. Every
config is simulated on its own trials, and the simulations are joined along the trial axis (`tasks.simulate_pooled`).
That is why the b-triple and lapse are profiled once, on all of a subject's trials.

## The likelihood (`likelihood/scoring.py`)

A simulation gives n_reps decisions and confidences per trial. The boundaries bin each confidence into a level
(`np.digitize`, right=False: level k holds b_k ≤ c < b_(k+1)). Per trial,

    p(d, r) = (1 − lapse) · count(d, r) / n_reps + lapse / 12,   floored at machine epsilon

and the NLL is −Σ log p(d_obs, r_obs).

## Profiling the boundaries and lapse (`likelihood/profile.py`)

Simulation is the expensive step, and re-scoring the same simulation at another (b1, b2, b3, lapse) is cheap. So
at every searched cell the model is simulated once, and the b-triple and lapse are minimised out:

1. An exhaustive seed over a 40-level grid of ordered b-triples × 8 lapse levels. It is approximate: confidence is
   histogrammed at 1/4096 resolution.
2. Nelder–Mead from the seed, against the exact objective.
3. The optimum is re-scored exactly and split into a decision term and a confidence-given-decision term.

The soft-attention models take one extra step (`tasks.profile`). Near β = ∞ their full-entropy confidence is the
two-category scale squeezed onto [1 − 1/log₂ 3, 1], and the b-triple search is not scale-invariant. So they are
profiled on the two-category scale first, and that optimum warm-starts the full-scale profile. With this, a soft
model never scores worse than its hard twin at a shared cell.

## Search (`fitting/search.py`, grids in `fitting/settings.py`)

| models | search |
|---|---|
| k = 6 (Diff, Ent, hard attention) | the full grid: 37 σ_x (0, then 10^−1 … 10^2.5) × 20 log₁₀ α (−0.5 … 4 step 0.25, and ∞) |
| k = 7 (soft attention) | coarse to fine. Stage A: every 2nd σ_x × 7 α × 8 log₁₀ β (−1 … 2 step 0.5, and ∞). Stage B: around each subject's 3 best stage-A cells, ±2 σ_x rungs × α ± 0.5 and β ± 0.25 at step 0.125 (β ∞ kept reachable; floor −1.25) |

The search uses 2,500 replicates per trial.

## Clean re-score (`fitting/fit.py`, `tasks.clean_score`)

The profiled NLL is fitted to the noise of one Monte Carlo draw. So each subject's 20 best search cells are:

1. re-profiled on a fresh draw of 10,000 replicates per trial (`refit_profile_nll`), then
2. scored at those boundaries and that lapse on another, independent draw of 100,000 (`nll`, the clean NLL).

The best fit of a subject is the cell with the lowest clean NLL, and AIC = 2 (k + NLL).

## Random numbers (`fitting/seeding.py`)

- **Search**: every cell of a subject uses the same stream per config, `default_rng([seed, subject, cfg])`. This
  common-random-numbers scheme makes the profiled surface smooth in the parameters.
- **Clean**: each cell gets a key `[seed, subject, sig_x_index, alpha word (, beta word)]`. The refit uses
  `key + [1, block]` and the clean NLL uses `key + [2, block]`, so the reported NLL comes from draws the boundaries were not
  fitted to.

The keys, the order of the draws inside `simulate` and the pinned numpy/scipy versions together make a fit
reproducible bit for bit. `scripts/check_paper_fits.py` uses this to confirm that the paper's fits come back exactly.

## Output (one folder per run, `results/fits/<model>_<date>_<time>/`)

| file | contents |
|---|---|
| `config_log.txt` | model, subjects, grids, seed, replicates, git commit, start and finish times |
| `profile.csv` | every search cell: σ_x, α (β), profiled NLL, b-triple, lapse (not tracked by git: large) |
| `clean.csv` | the shortlisted cells after the clean re-score |
| `fit_results.csv` | one row per subject: the best cell, `nll`, `k`, `aic`, `edge` flags |
| `report.md` | summed NLL and AIC, β summary (soft models), per-subject table |

`edge` lists the grid ends, and the nested values of β, that come within 2 nats of the subject's best search cell:
`alpha lo`, `alpha inf within 2`, `sig_x=0`, `sig_x hi`, `beta lo`, `beta=inf within 2`, `beta~1 within 2`.

## Running

    python scripts/fit_model.py --model sym_spec_soft             # 14 workers by default
    python scripts/fit_model.py --model diff --subjects 1 --simntrl 400 --smoke   # a quick end-to-end check
    python scripts/fit_all_models.py                              # all ten
    python scripts/summarise_fits.py                              # tabulate the newest runs

The paper's runs took 26–44 min per k = 6 model and about 1.5 h per soft-attention model.
