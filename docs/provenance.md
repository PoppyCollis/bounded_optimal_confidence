# Where the code came from

This package is a cut-down port of the paper's human-data fitting pipeline in `ITBR_confidence`. Only the code
paths the ten paper models use were carried over. The code was restructured and renamed, but the arithmetic and the
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

## Not carried over

These remain in `ITBR_confidence`: experiments 2 and 3; per-config fits and `--cfgs`; `--b-per-cfg`; the models
with a per-config choice (chirality, `ci_pick`); `--polish-all`; the hand-placed-cut models (`isi_*`, `cut*`);
`top2wt` / `diffwt` / γ; the BR and RD fits; and the MATLAB originals.

## Paper fits

`results/paper_fits/<model>/` holds the original runs' `fit_results.csv`, `clean.csv`, `report.md` and
`config_log.txt`, copied unchanged. `results/paper_fits/README.md` maps each one to its original folder and commit.
