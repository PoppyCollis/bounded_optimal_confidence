# bounded_optimal_confidence

Supporting code for "Decision confidence reflects bounded-optimal attention", (P Collis, P F Kinghorn, R Singh, P Lanillos and C L Buckley).

Code contains two parts: 

1. Model fits to human decision confidence data.
2. Information-theoretic bounded-rational analysis
   a) The bounded-optimal attention model optimised over single configurations of the experiment.
   b) The optimal shared attention mechanism over a prior over configurations.

## Install

    python3.14 -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev]"

The dependencies are pinned to the versions the paper's fits were run with. The fits are Monte Carlo with fixed seeds,
so they come back bit for bit only on the same numpy and scipy.

## Layout

    src/bounded_optimal_confidence/
        experiment.py       task constants and the trial data
        models/             bayes (posterior), attention (repertoires, collapsed inference), readouts,
                            registry (the ten models), simulate
        likelihood/         scoring (p(decision, level), clean NLL), profile (boundaries and lapse minimised out)
        fitting/            settings (grids), seeding (random streams), tasks (worker jobs), search, fit, report
        runs.py             reading fit results
        stats/              bms (Bayesian model selection), paired (t-test, Wilcoxon), bootstrap, tables
        plotting/           style, model_bars (the shared bar layout), bms_figures
        bounded_rational/   the hierarchy: blahut_arimoto (the solver), world_model and world (U and p(w) per layout),
                            seeds, reduced (router-pinned solve), identity, score, info_terms, rate_distortion,
                            candidates (top-2 pathway, specialist solve), transfer, rate_utility (the sweep), paper_run
        shared_attention/   optimise_attention_discrete (sparse structure priors and the shared router over sampled
                            worlds, the B sweep), paper_run
        utils/              worker pool, git provenance
    scripts/                the runnable entry points above
    paper_plots/            the paper's figure scripts and their figures (paper_plots/figures; runs/ and smoke/: from new runs)
    tests/
    data/li_ma_2020/        the experiment 1 data, with its README
    results/paper_fits/     the fits reported in the paper, one folder per model (README: provenance)
    results/paper_rate_utility/   the sweep the rate/cost-utility figure is drawn from (README: provenance, columns)
    results/paper_shared_attention/   the B sweep the p(x|phi) figure is drawn from (README: model, table, columns)
    results/fits/, results/rate_utility/, results/shared_attention/   new runs

## Reproducing the paper's runs

The three full runs, as re-run on 2026-10-25 (wall times on a 20-core machine in brackets):

    python scripts/fit_all_models.py --n-workers 14                                   # the ten model fits (9.5 h)
    python scripts/rate_utility_sweep.py --workers 14                                 # the rate-utility sweep (45 min)
    python scripts/optimise_attention_discrete.py --world-prior recentred --sweep-B 64 125 250 500 1000 2000 --n-seeds 5 --n-workers 2
                                                                                      # the shared-attention B sweep (52 min)

## The paper's fits

Bayesian model selection over the ten (protected exceedance probability): Diff 0.47, Sym Top-2 (soft) 0.51, every
other model ≤ 0.006, omnibus risk 0.0015. Neither Sym Top-2 model differs from Diff on a paired t-test of the
per-subject AIC (hard: t(12) = 1.19, p = 0.26; soft: t(12) = −0.90, p = 0.39); every other model is worse,
p < 0.001. 

## Acknowledgements

The three-variable Blahut-Arimoto solver in `bounded_rational/blahut_arimoto.py` is based on the three-variable
optimisation in Tim Genewein's
[BoundedRationalityAbstractionAndHierarchicalDecisionMaking](https://github.com/tgenewein/BoundedRationalityAbstractionAndHierarchicalDecisionMaking)
(Genewein, Leibfried, Grau-Moya and Braun, 2015, "Bounded rationality, abstraction, and hierarchical
decision-making: an information-theoretic optimality principle", *Frontiers in Robotics and AI* 2:27).

## Data

Li, H.-H. and Ma, W. J. (2020). Confidence reports in decision-making with multiple alternatives violate the Bayesian
confidence hypothesis. *Nature Communications* 11, 2004. See [data/li_ma_2020/README.md](data/li_ma_2020/README.md).
