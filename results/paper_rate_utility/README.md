# The paper's rate-utility sweep

`rateutil_20260922_144534/` is the run the rate-utility / cost-utility figure is drawn from
(`paper_plots/fig_rate_cost_utility.py`), copied unchanged from
`ITBR_confidence/generalisation/results/rateutil_20260922_144534` (made by `generalisation.rate_utility_beta3` on
2026-09-22 at commit 6c87f79-dirty; the four quick-look PNGs of the original folder are not copied). The same sweep is
`python scripts/rate_utility_sweep.py` here.

| | |
|---|---|
| axis | beta3: 80 rungs, geometric over [0.05, 100], with beta3 = 1 folded in exactly; beta1 = beta2 = inf; q = 0.5; two model states |
| worlds x configs | raw and norm x configs 1-4 = 8 worlds; 640 cells; 1920 rows (three families per cell) |
| families | `rd` (1-step rate distortion, cold-started at every beta3), `top2` (the pinned symmetric top-2 hierarchy), `solved` (the bounded-optimal card-2 hierarchy: best of three structured seeds, collapsed branch excluded) |
| solve | eps_conv 1e-8 held 300 iterations, cap 50000, alpha 0.05; 2600.8 s on 12 workers |
| outcome | all 640 solved cells stopped on `diff` at cardinality 2; no non-finite rows; winning seeds `cut_d1` 392, `cut_d2` 194, `midsplit` 54; self-transfer residual max 1.2e-9; the collapsed branch would have won in 0 of 640 |

Files: `curves.csv` (the rows), `config_log.txt` (axis, worlds, the original config dumps, run summary, checks: the
first `hard_router` check used log 2 instead of H(p(o)) and reads `ok False`; the re-run at the bottom uses the
corrected check and reads `ok True`), `summary.csv` / `summary.md` (the original quick-look tables).

`python scripts/check_paper_rate_utility.py` re-solves cells with this package and compares every column with
`curves.csv` exactly (`--all` for the whole sweep); `pytest -m slow -k bounded_rational` re-solves two of them.

## Columns of `curves.csv`

| column | meaning |
|---|---|
| `world`, `stimulus_config`, `beta1`, `beta2`, `beta3`, `q`, `family` | the cell and the family |
| `E_U` | expected utility (the figures' y axis, "Accuracy") |
| `J` | `E_U - I_WO/beta1 - I_AO/beta2 - I_WAgO/beta3` from the true information terms (for `rd`: `E_U - I_WA/beta3`) |
| `I_WO`, `I_AO`, `I_WAgO`, `I_WA`, `I_AOgW` | I(W;O), I(A;O), I(W;A|O), I(W;A), I(A;O|W) in nats, from the joint p(a, o, w) |
| `cost_cond`, `cost_total` | `I_WAgO / beta3` (the cost-utility x axis) and `I_WA / beta3` |
| `n_actions_used` | actions with max_w p(a|w) > 1e-3 |
| `regridded`, `finite` | whether the router was interpolated onto another grid (never here); whether every term is finite |
| `self_transfer_resid`, `J_solver` | solved rows: |J re-scored through the joint - J at the solve|, and the solver's J |
| `seed`, `stop_reason`, `n_iters`, `diff_final`, `effective_cardinality`, `chirality`, `J_spread_seeds` | solved rows: the winning seed and its solve record; the J spread across the three seeds |
| `J_collapsed` | solved rows: J of the collapsed (p(o|w) = p(o)) branch at this beta3, excluded from the solve |
