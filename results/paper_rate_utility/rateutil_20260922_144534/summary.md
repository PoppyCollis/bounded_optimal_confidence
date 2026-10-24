# Rate-utility of the top-2 code in the BR architecture

Run `rateutil_20260922_144534`. beta1 = beta2 = inf, card 2, pinned `p(o|w)` = hard cut at w = 0 and `p(a|o) = [[q,0],[1-q,1-q],[0,q]]` at q = 0.5. beta3 is the only free parameter of the pinned family.

At beta1 = beta2 = inf both serial charges vanish, so **J = E[U] - I(W;A|X)/beta3** exactly, for both families.

## Figures

- `rate_utility_norm.png`
- `cost_utility_norm.png`
- `rate_utility_raw.png`
- `cost_utility_raw.png`

## Table

`J_gap_*` is top-2 minus solved at matched beta3, so it is <= 0 wherever the solved hierarchy is doing its job; `beta3_at_smallest_gap` is where the pinned code comes closest to it. `EU_gap_max` is the same comparison on UTILITY alone, which is the half that is not foregone: the pinned code can buy more utility than the optimum and still lose on J, by paying more rate for it.

| world | config | top2_EU_b3_1 | top2_J_b3_1 | top2_J_best | top2_beta3_at_best_J | top2_worst_dJ | solved_EU_b3_1 | solved_J_b3_1 | solved_J_best | solved_beta3_at_best_J | solved_worst_dJ | J_gap_max | J_gap_min | beta3_at_smallest_gap | EU_gap_max | n_beta3_top2_EU_wins |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| norm | 1 | -6.6415 | -6.7658 | -6.4559 | 100.0000 | 0.0000 | -6.6337 | -6.6337 | -6.4543 | 100.0000 | -0.0001 | -0.0017 | -0.3964 | 100.0000 | 0.0417 | 5 |
| norm | 2 | -6.6778 | -6.8967 | -6.5000 | 100.0000 | 0.0000 | -6.6475 | -6.7993 | -6.4982 | 100.0000 | 0.0000 | -0.0017 | -0.5653 | 100.0000 | 0.0683 | 4 |
| norm | 3 | -6.6512 | -6.7237 | -6.5167 | 100.0000 | 0.0000 | -6.6169 | -6.6169 | -6.5157 | 100.0000 | -0.0000 | -0.0011 | -0.5173 | 100.0000 | -0.0000 | 0 |
| norm | 4 | -6.6512 | -6.7237 | -6.5167 | 100.0000 | 0.0000 | -6.6169 | -6.6169 | -6.5157 | 100.0000 | -0.0000 | -0.0011 | -0.5173 | 100.0000 | -0.0000 | 0 |
| raw | 1 | -5.4071 | -5.5132 | -5.2072 | 100.0000 | 0.0000 | -5.3697 | -5.3697 | -5.2056 | 100.0000 | -0.0000 | -0.0016 | -0.3615 | 100.0000 | 0.0342 | 3 |
| raw | 2 | -5.4396 | -5.6439 | -5.2512 | 100.0000 | 0.0000 | -5.3998 | -5.5476 | -5.2494 | 100.0000 | -0.0000 | -0.0017 | -0.4702 | 100.0000 | 0.0514 | 2 |
| raw | 3 | -5.4072 | -5.4826 | -5.2700 | 100.0000 | 0.0000 | -5.3458 | -5.3572 | -5.2686 | 100.0000 | -0.0000 | -0.0014 | -0.5531 | 100.0000 | -0.0000 | 0 |
| raw | 4 | -5.4072 | -5.4826 | -5.2700 | 100.0000 | 0.0000 | -5.3458 | -5.3572 | -5.2686 | 100.0000 | -0.0000 | -0.0014 | -0.5531 | 100.0000 | -0.0000 | 0 |

`*_beta3_at_best_J` is the top of the swept axis in every row, because J rises with beta3 for both families -- the 1/beta3 charge vanishes while E[U] saturates. It is an axis end, not an interior optimum. `*_worst_dJ` is the worst step down along the axis, so a real turnover would show there.

## Reading the marker

The ringed point on every curve is beta3 = 1, where the pinned code is hard top-2 Bayes exactly: with beta2 = inf the decision layer is `p(a|o) x exp(beta3 U)` and `exp(U)` is proportional to `q(w|a)`, so `p(a|o)` enters as a prior and beta3 = 1 is Bayes' rule on the surviving pair. Any other beta3 is a power tilt on the pair odds.

