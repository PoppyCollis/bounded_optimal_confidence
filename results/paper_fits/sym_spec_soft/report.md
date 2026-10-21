# entropy_ci_sym_beta_alpha_sigx_pooled, exp 1, cfgs [1, 2, 3, 4] pooled -- entropy_ci_sym_beta_alpha_sigx_pooled_cfg1234_20261006_061032

## fit

- summed NLL **6473.4**, AIC **13128.9** over 13 subjects, k = 7
- no per-config run for every cfg; pooling cost not computed
- beta: inf (hard selection) for 0/13; finite fits median **1.33**, range [1, 1.78]
- within 2 nats: beta = inf for 0/13, beta = 1 for 1/13, the grid bottom (beta 0.0562) for 0/13
- vs pooled ci_sym (nested, beta = inf):
- vs k = 6 `entropy_ci_sym_alpha_sigx_pooled_cfg1234_20261006_001634` on 13 subjects: summed delta NLL **-3575.9**, delta AIC **-7125.8** (negative favours this model); beta wins on AIC for 13/13
  - worse than that nested fit by > 1 nat (search failure): none
- vs pooled entropy (nested, beta = 1: the Bayes posterior):
- vs k = 6 `entropy_alpha_sigx_pooled_cfg1234_20261004_111153` on 13 subjects: summed delta NLL **-275.1**, delta AIC **-524.1** (negative favours this model); beta wins on AIC for 12/13
  - worse than that nested fit by > 1 nat (search failure): none
- vs pooled top2hard (k 6; AICs only):
- vs k = 6 `entropy_top2_hard_alpha_sigx_pooled_cfg1234_20260926_193133` on 13 subjects: summed delta NLL **+675.1**, delta AIC **+1376.2** (negative favours this model); ci_sym_beta wins on AIC for 0/13
- vs pooled diff (k 6; AICs only):
- vs k = 6 `diff_alpha_sigx_pooled_cfg1234_20260928_151314` on 13 subjects: summed delta NLL **+693.5**, delta AIC **+1412.9** (negative favours this model); ci_sym_beta wins on AIC for 0/13


```
 subject  sig_x  log10_alpha  log10_beta    b1    b2    b3  lapse     nll      aic                    edge
       1  0.316        0.000       0.250 0.404 0.772 0.994  0.000 522.865 1059.730                        
       2  5.012        0.500       0.250 0.398 0.563 0.920  0.000 480.916  975.831                        
       3  0.158        0.000       0.250 0.432 0.903 1.000  0.000 535.260 1084.521                        
       4  0.794        0.750       0.125 0.107 0.426 0.701  0.021 486.078  986.156                        
       5  0.126        0.500       0.125 0.393 0.559 0.874  0.000 524.745 1063.491                        
       6  0.200        0.375       0.125 0.388 0.666 0.998  0.004 514.730 1043.460                        
       7  1.995        0.875       0.125 0.133 0.389 0.598  0.010 471.553  957.106                 sig_x=0
       8  0.126        0.500       0.000 0.027 0.388 0.871  0.000 444.498  902.996 sig_x=0,beta~1 within 2
       9  0.200        0.375       0.125 0.415 0.677 0.983  0.000 532.010 1078.020                        
      10  7.943        1.000       0.250 0.391 0.523 0.748  0.000 420.595  855.189                        
      11  5.012        0.250       0.250 0.392 0.609 0.998  0.000 488.023  990.045                        
      12  0.794        0.500       0.125 0.393 0.503 0.745  0.000 538.218 1090.436                 sig_x=0
      13 15.849        1.000       0.125 0.208 0.398 0.756  0.021 513.943 1041.886                        
```

