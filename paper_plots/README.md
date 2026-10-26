# paper_plots

The paper's figure scripts, run from the repository root; each writes its PNG and SVG to `paper_plots/figures/`.

| script | figure | reads |
|---|---|---|
| `fig_likelihood_confidence.py [--norm data\|theory]` | `fig_likelihood_confidence` (2AFC and 3AFC: the likelihoods and the Entropy / Max / Difference / Entropy Top2 readouts of the posterior, each scaled to [0, 1] over the sweep; `--norm theory` scales by the readout's theoretical range and writes `fig_likelihood_confidence_theory`) | nothing |
| `fig_decision_confidence_panels.py` | `decision_confidence_panels` (decision and confidence of the Difference and Entropy Top-2 models on a bar plot of option probabilities) | nothing |
| `fig_config_likelihoods.py` | `fig_config_likelihoods` (the dot stimulus of layout 1 and the four layouts' likelihoods; the output depends very slightly on Python's hash seed, see the script) | nothing |
| `fig_pooled_aic.py` | `pooled_aic_bars` (summed ΔAIC of every model against Diff with bootstrap 95% CIs) | `results/paper_fits/<model>/fit_results.csv` |
| `fig_rate_cost_utility.py [--cfg 1\|2\|3\|4\|all] [--world norm\|raw] [--run DIR]` | `rate_cost_utility_cfg1_norm` (two panels: rate-utility, cost-utility; one layout) and `rate_cost_utility_all_norm` (4 x 2, one row per layout) | `results/paper_rate_utility/rateutil_20261025_141721/curves.csv`, or a new sweep's folder via `--run` |
| `fig_pxgphi_vs_B.py [--run DIR \| --latest]` | `pxgphi_vs_B_recentred_discrete` (p(x|phi_1), p(x|phi_2) and the router's switch-point against B/sigma) | `results/paper_shared_attention/optimise_attention_discrete_Bsweep_recentred_n10000_sig64_20261025_150919/sweep.csv`, or a new sweep via `--run` / `--latest` |

The BMS table (`bms_table`) and PXP bar (`bms_pxp_bars`) figures of the statistics are written here too, by
`scripts/plot_bms.py --paper` (reads `results/paper_fits/<model>/fit_results.csv`); without `--paper` it writes the newest
runs' figures to `figures/runs/`, with `--smoke` to `figures/smoke/`.

The shared pieces live in the package: `plotting/style.py` (palette, text sizes, rcParams), `plotting/readout_traces.py` (the 1D posterior and its readouts for the explanatory figures) and `plotting/model_bars.py` (the one-bar-per-model layout).
