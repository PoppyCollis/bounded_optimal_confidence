# paper_plots

The paper's figure scripts, run from the repository root; each writes its PNG and SVG to `paper_plots/figures/`.

| script | figure | reads |
|---|---|---|
| `fig_rate_cost_utility.py [--cfg 1\|2\|3\|4\|all] [--world norm\|raw] [--run DIR]` | `rate_cost_utility_cfg1_norm` (two panels: rate-utility, cost-utility; one layout) and `rate_cost_utility_all_norm` (4 x 2, one row per layout) | `results/paper_rate_utility/rateutil_20260922_144534/curves.csv`, or a new sweep's folder via `--run` |
| `fig_pxgphi_vs_B.py [--run DIR \| --latest]` | `pxgphi_vs_B_recentred_discrete` (p(x|phi_1), p(x|phi_2) and the router's switch-point against B/sigma) | `results/paper_shared_attention/optimise_attention_discrete_Bsweep_recentred_n10000_sig64_20261012_150128/sweep.csv`, or a new sweep via `--run` / `--latest` |

The BMS table and PXP figures of the statistics live under `figures/paper/` (`scripts/plot_bms.py --paper`).
