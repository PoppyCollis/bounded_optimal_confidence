# bounded_optimal_confidence

The paper's model fits to human confidence reports. The data are experiment 1 of Li & Ma (2020): 13 subjects categorise a
dot into one of three categories and rate their confidence on four levels, across four category layouts.

Ten decision models are fitted. All of them simulate a noisy measurement, a category posterior, Dirichlet decision noise,
an argmax decision and a confidence readout. They differ in what happens to the posterior before the decision:

- **Diff** and **Ent**: the standard Bayes posterior, read out as top-1 minus top-2 (Diff) or as entropy (Ent).
- **Attention models**: the posterior comes from collapsed inference over a repertoire of attentional masks.
  The four repertoires are Sym Top-2, Sym Spec, Asym-L and Asym-R. Each is fitted with **hard** attention (the
  maximum-evidence mask, β = ∞) and with **soft** attention (β fitted).

[docs/models.md](docs/models.md) gives the models in full, and [docs/fitting.md](docs/fitting.md) describes the fit.

## Install

    python3.14 -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev]"

The dependencies are pinned to the versions the paper's fits were run with. The fits are Monte Carlo with fixed seeds,
so they come back bit for bit only on the same numpy and scipy.

## Run

| command | what it does |
|---|---|
| `python scripts/fit_model.py --model sym_spec_soft` | fit one model to all 13 subjects → `results/fits/<model>_<date>_<time>/` |
| `python scripts/fit_model.py --model diff --subjects 1 --simntrl 400 --smoke` | a quick end-to-end check (about a minute) |
| `python scripts/fit_all_models.py` | fit all ten (about 10 h) |
| `python scripts/summarise_fits.py [--paper]` | summed NLL / AIC, ΔAIC to Diff, nested-model search checks |
| `python scripts/check_paper_fits.py` | re-score the paper's fits and confirm they are reproduced exactly (about 7 min) |
| `pytest` (`pytest -m slow` for the re-score tests) | the model identities, the likelihood, the paper fits |

The model names are `diff`, `ent`, `sym_top2_hard`, `sym_spec_hard`, `asym_spec_l_hard`, `asym_spec_r_hard`,
`sym_top2_soft`, `sym_spec_soft`, `asym_spec_l_soft` and `asym_spec_r_soft`. Scripts use 14 worker processes by
default; change it with `--n-workers`.

## Layout

    src/bounded_optimal_confidence/
        experiment.py       task constants and the trial data
        models/             bayes (posterior), attention (repertoires, collapsed inference), readouts,
                            registry (the ten models), simulate
        likelihood/         scoring (p(decision, level), clean NLL), profile (boundaries and lapse minimised out)
        fitting/            settings (grids), seeding (random streams), tasks (worker jobs), search, fit, report
        runs.py             reading fit results
        utils/              worker pool, git provenance
    scripts/                the runnable entry points above
    tests/
    data/li_ma_2020/        the experiment 1 data, with its README
    results/paper_fits/     the fits reported in the paper, one folder per model (README: provenance)
    results/fits/           new runs
    docs/                   models, fitting, provenance (how this maps to the original ITBR_confidence code)

## The paper's fits

| model | k | summed NLL | summed AIC | ΔAIC vs Diff |
|---|---|---|---|---|
| Diff | 6 | 5780.0 | 11715.9 | 0 |
| Ent | 6 | 6748.5 | 13653.0 | +1937.0 |
| Sym Top-2 (hard) | 6 | 5798.3 | 11752.7 | +36.7 |
| Sym Spec (hard) | 6 | 10049.4 | 20254.7 | +8538.8 |
| Asym-L (hard) | 6 | 8842.8 | 17841.6 | +6125.7 |
| Asym-R (hard) | 6 | 8681.2 | 17518.3 | +5802.4 |
| Sym Top-2 (soft) | 7 | 5754.1 | 11690.2 | −25.7 |
| Sym Spec (soft) | 7 | 6473.4 | 13128.9 | +1412.9 |
| Asym-L (soft) | 7 | 6513.7 | 13209.4 | +1493.4 |
| Asym-R (soft) | 7 | 6517.6 | 13217.1 | +1501.2 |

## Data

Li, H.-H. and Ma, W. J. (2020). Confidence reports in decision-making with multiple alternatives violate the Bayesian
confidence hypothesis. *Nature Communications* 11, 2004. See [data/li_ma_2020/README.md](data/li_ma_2020/README.md).
