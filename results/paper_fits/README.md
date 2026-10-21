# The paper's fits

The fits reported in the paper: one folder per model, holding the files of the original run in `ITBR_confidence`
(`human_data_model_fitting/results/<run>`), copied unchanged. They were made by `human_data_model_fitting.fit_alpha_pooled
--model <old name>` (configs 1-4 pooled, the default). The search grid (`profile.csv`, 1-3 MB each) is not copied.

| model | paper label | k | summed NLL | summed AIC | original run folder | old `--model` | | date |
|---|---|---|---|---|---|---|---|---|
| `diff` | Diff | 6 | 5780.0 | 11715.9 | `diff_alpha_sigx_pooled_cfg1234_20260928_151314` | `diff` | 9bed53a-dirty | 2026-09-28 |
| `ent` | Ent | 6 | 6748.5 | 13653.0 | `entropy_alpha_sigx_pooled_cfg1234_20261004_111153` | `entropy` | a191964-dirty | 2026-10-04 |
| `sym_top2_hard` | Sym Top-2 (hard) | 6 | 5798.3 | 11752.7 | `entropy_top2_hard_alpha_sigx_pooled_cfg1234_20260926_193133` | `top2hard` | d9ce659-dirty | 2026-09-26 |
| `sym_spec_hard` | Sym Spec (hard) | 6 | 10049.4 | 20254.7 | `entropy_ci_sym_alpha_sigx_pooled_cfg1234_20261006_001634` | `ci_sym` | 9433f5d-dirty | 2026-10-06 |
| `asym_spec_l_hard` | Asym-L (hard) | 6 | 8842.8 | 17841.6 | `entropy_ci_asym_left_alpha_sigx_pooled_cfg1234_20261005_224936` | `ci_asym_left` | 6cc0cf1-dirty | 2026-10-05 |
| `asym_spec_r_hard` | Asym-R (hard) | 6 | 8681.2 | 17518.3 | `entropy_ci_asym_right_alpha_sigx_pooled_cfg1234_20261005_233324` | `ci_asym_right` | 9433f5d-dirty | 2026-10-05 |
| `sym_top2_soft` | Sym Top-2 (soft) | 7 | 5754.1 | 11690.2 | `entropy_ci_top2_beta_alpha_sigx_pooled_cfg1234_20261006_074648` | `ci_top2_beta` | 9433f5d-dirty | 2026-10-06 |
| `sym_spec_soft` | Sym Spec (soft) | 7 | 6473.4 | 13128.9 | `entropy_ci_sym_beta_alpha_sigx_pooled_cfg1234_20261006_061032` | `ci_sym_beta` | 9433f5d-dirty | 2026-10-06 |
| `asym_spec_l_soft` | Asym-L (soft) | 7 | 6513.7 | 13209.4 | `entropy_ci_asym_left_beta_alpha_sigx_pooled_cfg1234_20261006_030532` | `ci_asym_left_beta` | 9433f5d-dirty | 2026-10-06 |
| `asym_spec_r_soft` | Asym-R (soft) | 7 | 6517.6 | 13217.1 | `entropy_ci_asym_right_beta_alpha_sigx_pooled_cfg1234_20261006_043801` | `ci_asym_right_beta` | 9433f5d-dirty | 2026-10-06 |

`python scripts/check_paper_fits.py` re-scores every subject's best cell with this package's code and compares it
with the stored NLL; `python scripts/summarise_fits.py --paper` tabulates the fits.

## Columns of the original files

`fit_results.csv` (one row per subject) uses the original names, which differ from new runs in a few columns:

| original | meaning | in new runs |
|---|---|---|
| `pid` | index of `sig_x` on the grid (`fitting.settings.SIG_X_GRID`) | `sig_x_index` |
| `t`, `chirality`, `rule` | per-config options of other models; constant here (`0`, `LLLL`, `-`) | dropped |
| `model_name` | the original run name | `model` (the new name) |

The other columns are the same: `sig_x`, `log10_alpha`, `log10_beta` (soft attention only; `inf` = hard), `b1`-`b3`
(confidence boundaries), `lapse`, `refit_profile_nll` (the profiled NLL of the refit draw), `nll` (the clean NLL,
100,000 replicates per trial), `edge` (grid-edge flags, see `fitting.report.edge_flags`), `k` and `aic`.
`clean.csv` holds every shortlisted cell after its clean re-score, and `report.md` and `config_log.txt` are the
original run's report and settings. The original reports also compare against the other runs that existed at the time.
