# The paper's fits

The fits reported in the paper: one folder per model, made by this package on 2026-10-25
(`python scripts/fit_all_models.py --n-workers 14`, 9.5 h for the ten; configs 1-4 pooled, one parameter set per
subject). Each folder holds the run's `fit_results.csv`, `clean.csv`, `report.md` and `config_log.txt`; the search
grid (`profile.csv`, 1-3 MB) is kept on disk but not tracked.

| model | paper label | k | summed NLL | summed AIC | run | commit |
|---|---|---|---|---|---|---|
| `diff` | Diff | 6 | 5780.0 | 11715.9 | `diff_20261025_151122` | `43e2e06` |
| `ent` | Ent | 6 | 6748.5 | 13653.0 | `ent_20261025_153721` | `43e2e06-dirty` |
| `sym_top2_hard` | Sym Top-2 (hard) | 6 | 5798.3 | 11752.7 | `sym_top2_hard_20261025_160440` | `43e2e06-dirty` |
| `sym_spec_hard` | Sym Spec (hard) | 6 | 10049.4 | 20254.7 | `sym_spec_hard_20261025_163916` | `43e2e06-dirty` |
| `asym_spec_l_hard` | Asym-L (hard) | 6 | 8842.8 | 17841.6 | `asym_spec_l_hard_20261025_171828` | `43e2e06-dirty` |
| `asym_spec_r_hard` | Asym-R (hard) | 6 | 8681.2 | 17518.3 | `asym_spec_r_hard_20261025_175809` | `43e2e06-dirty` |
| `sym_top2_soft` | Sym Top-2 (soft) | 7 | 5754.1 | 11690.2 | `sym_top2_soft_20261025_183803` | `43e2e06-dirty` |
| `sym_spec_soft` | Sym Spec (soft) | 7 | 6473.4 | 13128.9 | `sym_spec_soft_20261025_200959` | `bfab47d-dirty` |
| `asym_spec_l_soft` | Asym-L (soft) | 7 | 6513.7 | 13209.4 | `asym_spec_l_soft_20261025_214106` | `bfab47d-dirty` |
| `asym_spec_r_soft` | Asym-R (soft) | 7 | 6517.6 | 13217.1 | `asym_spec_r_soft_20261025_231156` | `bfab47d-dirty` |

`python scripts/check_paper_fits.py` re-scores every subject's best cell with this package's code and compares it
with the stored NLL; `python scripts/summarise_fits.py --paper` tabulates the fits.

## Columns

`fit_results.csv` (one row per subject): `subject`, `sig_x_index` (index of `sig_x` on `fitting.settings.SIG_X_GRID`),
`sig_x`, `log10_alpha`, `log10_beta` (soft attention only; `inf` = hard), `b1`-`b3` (confidence boundaries), `lapse`,
`refit_profile_nll` (the profiled NLL of the refit draw), `nll` (the clean NLL, 100,000 replicates per trial), `edge`
(grid-edge flags, see `fitting.report.edge_flags`), `model`, `k` and `aic`. `clean.csv` holds every shortlisted cell
after its clean re-score, with the same columns up to `lapse`; `report.md` and `config_log.txt` are the run's report
and settings.
