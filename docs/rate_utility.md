# The rate-utility and cost-utility curves

Code: `src/bounded_optimal_confidence/bounded_rational/`. Scripts: `scripts/rate_utility_sweep.py` (the sweep),
`scripts/check_paper_rate_utility.py` (reproduce the paper's run), `paper_plots/fig_rate_cost_utility.py` (the figure).
The paper's run is `results/paper_rate_utility/rateutil_20260922_144534/` (its README lists the columns).

## The objective

A bounded-rational decision maker sees a world state w (the stimulus position on a line), picks a model state o
(an attentional state, two of them) and then an action a (one of three categories), paying for information at
three exchange rates:

    J = E[U] - I(W;O)/beta1 - I(A;O)/beta2 - I(W;A|O)/beta3

with U(a, w) = log q(w|a) the category log-likelihood. The fixed points of this objective are found by the
three-variable Blahut-Arimoto iteration over p(o|w), p(a|o) and p(a|o,w) (`blahut_arimoto.threevarBAiterations`,
Genewein et al. 2015). The sweep holds beta1 = beta2 = inf, so both serial charges vanish and

    J = E[U] - I(W;A|O)/beta3

for every hierarchy, whatever its p(o|w) and p(a|o). That is what makes the cost-utility panel readable: its x axis is
the charged cost I(W;A|O)/beta3 for the hierarchies and I(W;A)/beta for one-step rate distortion, so J = E[U] - x on
every curve and iso-J lines have slope 1. The rate panel's x axis is the total rate I(W;A), which all three families
share.

## The three families

| family | what it is | how it is obtained |
|---|---|---|
| `rd` | one-step rate distortion: no model state, maximise E[U] - I(W;A)/beta over p(a|w) | `rate_distortion.onevar_ba`, cold-started from uniform p(a) at every beta, to max |dp(a)| < 1e-14. The frontier no policy can beat at its own rate. |
| `top2` | the symmetric top-2 hierarchy, pinned: p(o|w) a hard cut at w = 0 (beta1 = inf), p(a|o) = [[q, 0], [1-q, 1-q], [0, q]] at q = 0.5 (beta2 = inf); each model state drops one outer category and splits the kept pair evenly | `candidates.top2_pathway`; p(a|o,w) = p(a|o) exp(beta3 U) / Z is derived on each world (`rate_utility.lean_pagow`); at beta3 = 1 this is exactly the hard top-2 Bayes posterior |
| `solved` | the bounded-optimal hierarchy: the best two-state fixed point of the objective at (inf, inf, beta3) | `candidates.solve_specialist`: three structured seed routers (a cut at each Bayes boundary d1, d2, and the midsplit that pools the two tails), each completed by the reduced solve with the router pinned (`reduced.solve_reduced`, undamped, 1e-8 held 3 iterations), then the full BA (alpha 0.05, diff < 1e-8 held 300 iterations, cap 50000); the seed with the best J wins. The collapsed branch (p(o|w) = p(o), cardinality 1) is excluded from the competition and recorded beside the row as `J_collapsed`. |

Both hierarchies are scored the same way, through the joint p(a, o, w) = p(w) p(o|w) p(a|o,w)
(`info_terms.true_terms`). On a pinned policy the solver's own accounting reports the objective's charges rather than
mutual informations (0.185 against a true 0.075 nats on config 3), so scoring the two families by different routes
would put them in different units and the gap between the curves would mean nothing. The solved family is re-scored
through the same function, and the identity that this reproduces the J it was solved at is recorded per row
(`self_transfer_resid`; max 1.2e-9 in the paper's run).

## The worlds

Each of the four stimulus configurations of Li & Ma (2020) experiment 1 is a world: category means mu = [-96, 0, 96],
[-128, 0, 128], [-96, -59, 96] and [-96, 59, 96], sigma = 64. Two versions of each are swept:

| world | U(a, w) | p(w) | grid |
|---|---|---|---|
| `raw` | log N(w; mu_a, 64) | uniform | linspace(mu_1 - 64, mu_3 + 64, 1000) |
| `norm` | log q(w|a), q normalised over the grid | mean_a q(w|a) | 6 sigma past the outer means at the raw step 0.3203 (2998 points; 3198 on config 2) |

The paper's figure uses the normalised world (`--world norm`, the default), the one in which rate distortion at
beta = 1 is exactly the flat-prior Bayes posterior; `--world raw` draws the other.

## The sweep (`rate_utility.cell`)

beta3 runs over 80 geometric rungs in [0.05, 100] with beta3 = 1 folded in exactly (`rate_utility.beta3_axis`), on 8
worlds: 640 cells, three rows each. A cell is one task in a spawned worker pool, with BLAS pinned to one thread. Each
row carries E[U], J, the five information terms I(W;O), I(A;O), I(W;A|O), I(W;A), I(A;O|W), the two cost columns
`cost_cond = I(W;A|O)/beta3` and `cost_total = I(W;A)/beta3`, and for the solved family the winning seed, stop reason,
iteration count, effective cardinality and chirality.

After the sweep four checks run on the table and are written to `config_log.txt`:

- `hard_router`: on the top-2 family I(W;A) = I(A;O) + I(W;A|O) and I(A;O|W) = 0 exactly (a hard router adds no noise),
  and I(W;O) = H(p(o)), which is log 2 only where the cut halves p(w) (not on configs 3 and 4 in the norm world).
- `mirror_cfg3_cfg4`: configs 3 and 4 are each other's mirror, so their curves coincide (to 3e-10 for the solved
  family in the paper's run).
- `top2_is_bayes_at_beta3_1`: the pinned family at beta3 = 1 is the hard top-2 posterior to 2e-16.
- `lean_vs_transfer_score`: the masked scoring path equals the unmasked one (`transfer.score`) wherever the latter is
  finite. The mask exists because `seeds.state_from` builds p(a|o,w) for every (o, w), including the pairs the router
  gives zero weight, and out in the tail those underflow to 0/0 and poison the marginal.

`--recheck <run dir>` re-runs the checks on a finished run without solving.

## Running it

    python scripts/rate_utility_sweep.py                                           # the paper's sweep, ~40 min on 14 workers
    python scripts/rate_utility_sweep.py --tag smoke --configs 1 3 --worlds raw --beta3-n 6   # ~1 min
    python scripts/check_paper_rate_utility.py [--n N | --all]                      # reproduce stored cells exactly
    python paper_plots/fig_rate_cost_utility.py [--cfg 1|2|3|4|all] [--world norm|raw] [--run DIR]
    pytest tests/test_bounded_rational.py; pytest -m slow -k bounded_rational

The figure script reads `curves.csv` only. `--cfg 1` gives the two-panel figure of the paper
(`paper_plots/figures/rate_cost_utility_cfg1_norm.{png,svg}`); `--cfg all` the 4 x 2 grid over the configurations
(`rate_cost_utility_all_norm.{png,svg}`).

## What the paper's run found

The solved hierarchy beats the pinned top-2 code on J at every one of the 640 cells (it is optimal there and the other
is pinned); the gap narrows monotonically in beta3, from 0.10-0.14 nats at beta3 = 1 to 0.001-0.002 by beta3 = 100.
On utility alone the pinned code can exceed the optimum (configs 1 and 2 at some beta3), paying more rate for it. The
one-step rate-distortion frontier bounds both from above at every rate. All 640 solved cells stopped on `diff` at
cardinality 2 and the collapsed branch never won.
