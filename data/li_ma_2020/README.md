# Li & Ma (2020), experiment 1

Trial-by-trial data of the 13 participants of experiment 1 of

> Li, H.-H. and Ma, W. J. (2020). Confidence reports in decision-making with multiple alternatives violate the
> Bayesian confidence hypothesis. *Nature Communications* 11, 2004. doi:10.1038/s41467-020-15581-6

copied unchanged from the authors' repository [confidence-multiple-alternatives](https://github.com/hsinhungli/confidence-multiple-alternatives), folder `data/EXP1`). Please cite
the paper when using the data.

## Task

On each trial a target dot is drawn from one of three categories. Each category is a 2-D Gaussian of standard deviation
64 (screen units) around a category centre. The participant reports which category the dot came from and how confident they are, on 4 levels.
Experiment 1 uses four stimulus configurations, all collinear (every centre at y = 0):

| config | category centres (x) |
|---|---|
| 1 | -96, 0, 96 |
| 2 | -128, 0, 128 |
| 3 | -96, -59, 96 |
| 4 | -96, 59, 96 |

## Files

`EXP1/S<n>_log.mat`, n = 1..13. Each holds a MATLAB struct `trl` with one entry per trial:

| field | meaning |
|---|---|
| `config` | stimulus configuration (1-4) |
| `estD` | category decision (1-3, left to right) |
| `estC` | confidence report (1-4, low to high) |
| `target_x`, `target_y` | target dot location |
| `theta` | colour (angle on a colour wheel, radians) of the leftmost category; not used here |

They are read by `bounded_optimal_confidence.experiment.load_trials`.
