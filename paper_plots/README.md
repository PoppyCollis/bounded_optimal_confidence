# paper_plots

The paper's figure scripts, run from the repository root; each writes its PNG and SVG to `paper_plots/figures/`.

| script | figure | reads |
|---|---|---|
| `fig_rate_cost_utility.py [--cfg 1\|2\|3\|4\|all] [--world norm\|raw] [--run DIR]` | `rate_cost_utility_cfg1_norm` (two panels: rate-utility, cost-utility; one layout) and `rate_cost_utility_all_norm` (4 x 2, one row per layout) | `results/paper_rate_utility/rateutil_20260922_144534/curves.csv`, or a new sweep's folder via `--run` |

The BMS table and PXP figures of the statistics live under `figures/paper/` (`scripts/plot_bms.py --paper`).
