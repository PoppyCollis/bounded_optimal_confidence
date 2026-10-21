# entropy_ci_asym_left_beta_alpha_sigx_pooled, exp 1, cfgs [1, 2, 3, 4] pooled -- entropy_ci_asym_left_beta_alpha_sigx_pooled_cfg1234_20261006_030532

## fit

- summed NLL **6513.7**, AIC **13209.4** over 13 subjects, k = 7
- no per-config run for every cfg; pooling cost not computed
- beta: inf (hard selection) for 0/13; finite fits median **1.33**, range [1, 1.78]
- within 2 nats: beta = inf for 0/13, beta = 1 for 2/13, the grid bottom (beta 0.0562) for 0/13
- vs pooled ci_asym_left (nested, beta = inf):
- vs k = 6 `entropy_ci_asym_left_alpha_sigx_pooled_cfg1234_20261005_224936` on 13 subjects: summed delta NLL **-2329.1**, delta AIC **-4632.2** (negative favours this model); beta wins on AIC for 13/13
  - worse than that nested fit by > 1 nat (search failure): none
- vs pooled entropy (nested, beta = 1: the Bayes posterior):
- vs k = 6 `entropy_alpha_sigx_pooled_cfg1234_20261004_111153` on 13 subjects: summed delta NLL **-234.8**, delta AIC **-443.6** (negative favours this model); beta wins on AIC for 11/13
  - worse than that nested fit by > 1 nat (search failure): none
- vs pooled top2hard (k 6; AICs only):
- vs k = 6 `entropy_top2_hard_alpha_sigx_pooled_cfg1234_20260926_193133` on 13 subjects: summed delta NLL **+715.3**, delta AIC **+1456.7** (negative favours this model); ci_asym_left_beta wins on AIC for 0/13
- vs pooled diff (k 6; AICs only):
- vs k = 6 `diff_alpha_sigx_pooled_cfg1234_20260928_151314` on 13 subjects: summed delta NLL **+733.7**, delta AIC **+1493.4** (negative favours this model); ci_asym_left_beta wins on AIC for 0/13


```
 subject  sig_x  log10_alpha  log10_beta    b1    b2    b3  lapse     nll      aic                    edge
       1  0.251        0.250       0.250 0.384 0.621 0.945  0.000 508.906 1031.812                 sig_x=0
       2  0.251        0.500       0.125 0.392 0.524 0.859  0.000 513.382 1040.764                 sig_x=0
       3  0.501        0.250       0.250 0.401 0.765 0.986  0.000 531.284 1076.568                        
       4  2.512        0.750       0.125 0.084 0.426 0.691  0.046 485.492  984.985                        
       5  0.126        0.625       0.250 0.392 0.533 0.849  0.000 506.393 1026.785                        
       6  0.794        0.500       0.125 0.385 0.614 0.991  0.010 510.608 1035.216                 sig_x=0
       7  1.259        0.875       0.000 0.118 0.382 0.560  0.010 489.147  992.294 sig_x=0,beta~1 within 2
       8  0.251        0.500       0.125 0.051 0.397 0.905  0.000 443.707  901.414 sig_x=0,beta~1 within 2
       9  0.126        0.500       0.125 0.401 0.618 0.958  0.011 531.018 1076.035                        
      10  0.501        0.750       0.125 0.391 0.507 0.726  0.000 477.816  969.632                 sig_x=0
      11  0.316        0.250       0.250 0.394 0.601 0.998  0.000 477.273  968.547                        
      12  0.251        0.500       0.250 0.404 0.523 0.788  0.000 531.414 1076.827                 sig_x=0
      13  3.981        0.500       0.250 0.376 0.564 0.964  0.001 507.242 1028.483                        
```

