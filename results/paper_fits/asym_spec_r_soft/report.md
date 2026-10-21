# entropy_ci_asym_right_beta_alpha_sigx_pooled, exp 1, cfgs [1, 2, 3, 4] pooled -- entropy_ci_asym_right_beta_alpha_sigx_pooled_cfg1234_20261006_043801

## fit

- summed NLL **6517.6**, AIC **13217.1** over 13 subjects, k = 7
- no per-config run for every cfg; pooling cost not computed
- beta: inf (hard selection) for 0/13; finite fits median **1.33**, range [1, 1.78]
- within 2 nats: beta = inf for 0/13, beta = 1 for 1/13, the grid bottom (beta 0.0562) for 0/13
- vs pooled ci_asym_right (nested, beta = inf):
- vs k = 6 `entropy_ci_asym_right_alpha_sigx_pooled_cfg1234_20261005_233324` on 13 subjects: summed delta NLL **-2163.6**, delta AIC **-4301.2** (negative favours this model); beta wins on AIC for 13/13
  - worse than that nested fit by > 1 nat (search failure): none
- vs pooled entropy (nested, beta = 1: the Bayes posterior):
- vs k = 6 `entropy_alpha_sigx_pooled_cfg1234_20261004_111153` on 13 subjects: summed delta NLL **-230.9**, delta AIC **-435.8** (negative favours this model); beta wins on AIC for 13/13
  - worse than that nested fit by > 1 nat (search failure): none
- vs pooled top2hard (k 6; AICs only):
- vs k = 6 `entropy_top2_hard_alpha_sigx_pooled_cfg1234_20260926_193133` on 13 subjects: summed delta NLL **+719.2**, delta AIC **+1464.4** (negative favours this model); ci_asym_right_beta wins on AIC for 1/13
- vs pooled diff (k 6; AICs only):
- vs k = 6 `diff_alpha_sigx_pooled_cfg1234_20260928_151314` on 13 subjects: summed delta NLL **+737.6**, delta AIC **+1501.2** (negative favours this model); ci_asym_right_beta wins on AIC for 1/13


```
 subject  sig_x  log10_alpha  log10_beta    b1    b2    b3  lapse     nll      aic            edge
       1  0.251       -0.125       0.250 0.419 0.845 0.999  0.000 529.043 1072.085                
       2  0.631        0.500       0.250 0.399 0.543 0.892  0.000 497.878 1009.756         sig_x=0
       3  0.316        0.125       0.125 0.400 0.787 0.989  0.000 551.248 1116.497                
       4  1.585        0.750       0.125 0.085 0.432 0.705  0.044 489.971  993.942                
       5  0.251        0.500       0.125 0.394 0.562 0.887  0.000 524.696 1063.391                
       6  0.000        0.375       0.125 0.391 0.660 0.998  0.003 512.120 1038.240         sig_x=0
       7  0.200        0.875       0.125 0.131 0.394 0.614  0.021 477.912  969.823                
       8 12.589        0.500       0.250 0.052 0.410 0.963  0.000 419.568  853.136                
       9  1.585        0.500       0.125 0.399 0.620 0.965  0.011 525.110 1064.220                
      10  0.100        0.750       0.250 0.401 0.534 0.775  0.000 469.102  952.203         sig_x=0
      11  0.398        0.250       0.125 0.387 0.561 0.992  0.000 491.912  997.823                
      12  3.162        0.625       0.250 0.396 0.498 0.742  0.005 510.412 1034.824                
      13  7.943        0.875       0.000 0.188 0.393 0.704  0.009 518.590 1051.180 beta~1 within 2
```

