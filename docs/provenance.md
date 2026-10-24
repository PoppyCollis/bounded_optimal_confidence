# Where the code came from

This package is a cut-down port of the paper's code in `ITBR_confidence`: the human-data fitting pipeline, the
statistics on the fits, and the bounded-rational hierarchy sweep behind the rate/cost-utility figure. Only the code
paths the paper's runs use were carried over. The code was restructured and renamed, but the arithmetic and the
random streams were kept, so the port gives the same numbers bit for bit. This was checked in three ways during the port:

- `simulate`, the search task and the clean task were compared against the originals on random cells;
- the full pipeline was run through both codebases for all ten models in smoke mode (subject 1, 400 replicates per
  trial). `profile.csv`, `clean.csv` and `fit_results.csv` came out identical;
- the paper's fits were re-scored (`scripts/check_paper_fits.py`).

## Model names

| here | old `--model` (`fit_alpha_pooled`) | old run-folder prefix |
|---|---|---|
| `diff` | `diff` | `diff_alpha_sigx_pooled_cfg1234` |
| `ent` | `entropy` | `entropy_alpha_sigx_pooled_cfg1234` |
| `sym_top2_hard` | `top2hard` | `entropy_top2_hard_alpha_sigx_pooled_cfg1234` |
| `sym_spec_hard` | `ci_sym` | `entropy_ci_sym_alpha_sigx_pooled_cfg1234` |
| `asym_spec_l_hard` | `ci_asym_left` | `entropy_ci_asym_left_alpha_sigx_pooled_cfg1234` |
| `asym_spec_r_hard` | `ci_asym_right` | `entropy_ci_asym_right_alpha_sigx_pooled_cfg1234` |
| `sym_top2_soft` | `ci_top2_beta` | `entropy_ci_top2_beta_alpha_sigx_pooled_cfg1234` |
| `sym_spec_soft` | `ci_sym_beta` | `entropy_ci_sym_beta_alpha_sigx_pooled_cfg1234` |
| `asym_spec_l_soft` | `ci_asym_left_beta` | `entropy_ci_asym_left_beta_alpha_sigx_pooled_cfg1234` |
| `asym_spec_r_soft` | `ci_asym_right_beta` | `entropy_ci_asym_right_beta_alpha_sigx_pooled_cfg1234` |

The repertoires were renamed as well: `top2` → `sym_top2`, `sym` → `sym_spec`, `asym_left` → `asym_spec_l`,
`asym_right` → `asym_spec_r`.

## Modules

| here | from (`ITBR_confidence/...`) |
|---|---|
| `experiment.py` | `human_data_model_fitting/config.py` (`STI_PAR`, `N_SUBJECTS`), `data.py`, `fit_sampled_decision._trials` |
| `models/bayes.py` | `fit_sampled_decision.bayes_posterior` (its unused β tempering dropped) |
| `models/attention.py` | `implicit_structural_inference/collapsed_inference.py` (the model part), `nll._top2_posterior` |
| `models/readouts.py` | `nll._diff_confidence`, `_entropy_confidence`, `_entropy_top2_confidence` |
| `models/registry.py` | `fit_alpha_argmax_cfg1.K` / `READOUT`, `structure_cut.CI_HARD` / `CI_BETA` |
| `models/simulate.py` | `fit_alpha_argmax_cfg1.simulate`, `noisy_argmax`; `fit_sampled_decision.CHUNK_REPS` |
| `likelihood/scoring.py` | `nll.report_joint_probs`, `fit_sampled_decision.clean_nll` |
| `likelihood/profile.py` | `nll_profile.py`, without `profile_b_per_block`, `approx_joint_nll` and the diagnostics |
| `fitting/settings.py` | grids and constants from `fit_alpha_argmax_cfg1`, `fit_sampled_decision` and `config` |
| `fitting/seeding.py` | `fit_alpha_argmax_cfg1._alpha_key` / `_b_key` and the keys built inline in `fit_alpha_pooled` |
| `fitting/tasks.py` | `fit_alpha_pooled.rank_task`, `clean_task`, `_profile_model` |
| `fitting/search.py` | `fit_alpha_pooled.rank_grid` (k 6) and `rank_wt` (soft attention) |
| `fitting/fit.py` | `fit_alpha_pooled.main` (shortlist, clean, best) |
| `fitting/report.py` | `fit_alpha_argmax_cfg1.edge_flags`, `fit_alpha_pooled.beta_lines` |
| `utils/parallel.py` | `utils/parallel.py` |
| `stats/bms.py` | `paper_plots/fig_pooled_bms.fit` / `bms`, `pooled_bms_family.readout` / `family_prior`, and the `groupBMC` package's VB fit and exceedance probability, written out |
| `stats/bootstrap.py` | `paper_plots/bootstrap_ci.py` |
| `stats/paired.py` | new: the paired t-tests were ad hoc (recorded in `memory.md`) |
| `stats/tables.py` | `fig_pooled_bms.summary`, `pooled_bms_family.main`, `fig_pooled_aic.summary` (the bootstrap column) |
| `plotting/style.py` | `paper_plots/fig_likelihood_confidence.py` (palette, sizes, rcParams) |
| `plotting/model_bars.py` | `fig_pooled_aic.bar_figure` (the layout, legends) |
| `plotting/bms_figures.py` | `fig_pooled_bms.table_figure` / `bar_figure` |
| `bounded_rational/settings.py` | the constants the sweep reaches, from the root `config.py` (`ALPHA`, the patience counters, the cycle detector, `CARD_TOL`), `partition_enum/config.py` and `fixed_points/config.py` (`REDUCED_*`, `CHIRALITY_*`), `generalisation/config.py` (the world, `CONFIGS`, `WORLDS`, `EPS_CONV`, `FULL_MAXITER`) and `beta_sweep/norm_world.py` (`SIG_S`, `EPS_SD`, `GRID_STEP`); the environment-variable overrides dropped |
| `bounded_rational/helpers.py` | `utils/helpers.py`: `log_nats`, `boltzmanndist`, `boltzmanndist_vectorized`, `effective_cardinality_from_pago` |
| `bounded_rational/blahut_arimoto.py` | `three_var_blahut_arimoto.py`: the update functions, the information terms, `objective_value`, `_ba_step`, `_state_metrics`, the cycle detector, `threevarBAiterations` (the `*_old` loop versions dropped) |
| `bounded_rational/reduced.py` | `partition_enum/reduced_ba.py`: `marginalizeo_fast`, `_marginals_fast`, `hard_partition`, `solve_reduced` |
| `bounded_rational/world_model.py` | `beta_sweep/objective_task._world_model` (`mu_as` made a required argument: its default read a module global every caller overrides) and `beta_sweep/norm_world.py`: `normalised_world`, `raw_world`, `build` |
| `bounded_rational/rate_distortion.py` | `beta_sweep/objective_task.py`: `onevar_ba`, `collapsed_branch`, `_collapsed_branch_applies` |
| `bounded_rational/world.py` | `generalisation/world.py`: `build`, `mu_as_for` (reads `experiment.CATEGORY_X` instead of `human_data_model_fitting.config.STI_PAR`; same values), `same_grid`, `regrid_pogw`, `describe` |
| `bounded_rational/seeds.py` | `fixed_points/init_policy.py` (the structured seeds and the two completions; the random seeds dropped) and `two_state_attention/policies.state_from` |
| `bounded_rational/identity.py` | `fixed_points/identity.py`: `state_dict`, `is_mirror_symmetric`, `skew`, `crossing_centroid`, `chirality`; `regime_map/branches.crossings` |
| `bounded_rational/score.py` | `fixed_points/score.py` |
| `bounded_rational/info_terms.py` | `beta_sweep/info_terms.py`: `_entropy`, `rate_utility`, `true_terms` |
| `bounded_rational/candidates.py` | `generalisation/candidates.py`: `top2_pago`, `top2_pathway`, `solve_specialist` and its helpers |
| `bounded_rational/transfer.py` | `generalisation/transfer.py`: `objective`, `score` |
| `bounded_rational/rate_utility.py` | `generalisation/rate_utility_beta3.py`, whole: the axis, the masked scoring path, `cell`, the four checks, the config log and run summary, `main`. The run folder goes under `results/rate_utility/`, the config log dumps `settings` instead of three config modules, the git commit is this repository's |
| `bounded_rational/paper_run.py` | new: reading the stored run and comparing a re-solved cell with it |
| `scripts/rate_utility_sweep.py` | `python -m generalisation.rate_utility_beta3` |
| `paper_plots/fig_rate_cost_utility.py` | `paper_plots/fig_rate_cost_utility_cfg1.py`; the text sizes it imported from `bayes_1d_readouts` and `fig_likelihood_confidence` are declared in the script, the rcParams those set on import come from `plotting/style.py` |

## Not carried over

These remain in `ITBR_confidence`: experiments 2 and 3; per-config fits and `--cfgs`; `--b-per-cfg`; the models
with a per-config choice (chirality, `ci_pick`); `--polish-all`; the hand-placed-cut models (`isi_*`, `cut*`);
`top2wt` / `diffwt` / γ; the BR and RD fits; and the MATLAB originals.

From the bounded-rational side: the (beta2, beta3) and (beta1, beta3) grid sweeps of `beta_sweep/` and their readout
analyses; the transfer experiment, the shared-pathway experiments and `plot_rate_utility.py`'s quick-look figures in
`generalisation/`; the fixed-point store (`fixed_points/store.py`, `verify.py`); `regime_map`; `two_state_attention`;
`partition_enum`'s enumeration; random restarts; the beta3 = 0 (purely serial) branches, which the solver still
contains but no paper run enters.

## Paper fits

`results/paper_fits/<model>/` holds the original runs' `fit_results.csv`, `clean.csv`, `report.md` and
`config_log.txt`, copied unchanged. `results/paper_fits/README.md` maps each one to its original folder and commit.

## The rate-utility sweep

`results/paper_rate_utility/rateutil_20260922_144534/` is the original run, copied unchanged (its README gives the
provenance and the columns). The sweep is deterministic (structured seeds, no random numbers), so the port is checked
by re-solving cells: `scripts/check_paper_rate_utility.py` reproduces every column of the stored `curves.csv` exactly
on the cells it re-solves, and the same eight cells solved by the original code in its own environment gave the same
rows. The figure drawn from the stored run is pixel-identical to the original's PNG, and its SVG identical up to the
date stamp and matplotlib's random element ids. `docs/rate_utility.md` describes the sweep.

## Statistics

The Bayesian model selection and the bootstrap give the same numbers as the old `paper_plots` scripts on the same
input, to the last bit; the old scripts' default CSV float parsing differs from the stored values by one ulp on
some AICs, which moves the results by less than 1e-12. `docs/statistics.md` has the details, and the comparison
with the old notes' paired t-tests.
