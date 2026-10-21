# entropy_ci_top2_beta_alpha_sigx_pooled, exp 1, cfgs [1, 2, 3, 4] pooled -- entropy_ci_top2_beta_alpha_sigx_pooled_cfg1234_20261006_074648

## fit

- summed NLL **5754.1**, AIC **11690.2** over 13 subjects, k = 7
- no per-config run for every cfg; pooling cost not computed
- beta: inf (hard selection) for 2/13; finite fits median **42.2**, range [13.3, 178]
- within 2 nats: beta = inf for 5/13, beta = 1 for 0/13, the grid bottom (beta 0.0562) for 0/13
- vs pooled top2hard (nested, beta = inf):
- vs k = 6 `entropy_top2_hard_alpha_sigx_pooled_cfg1234_20260926_193133` on 13 subjects: summed delta NLL **-44.2**, delta AIC **-62.5** (negative favours this model); beta wins on AIC for 8/13
  - worse than that nested fit by > 1 nat (search failure): none
- vs pooled diff (k 6; AICs only):
- vs k = 6 `diff_alpha_sigx_pooled_cfg1234_20260928_151314` on 13 subjects: summed delta NLL **-25.9**, delta AIC **-25.7** (negative favours this model); ci_top2_beta wins on AIC for 6/13


```
 subject  sig_x  log10_alpha  log10_beta    b1    b2    b3  lapse     nll      aic              edge
       1 15.849        1.750         inf 0.371 0.402 0.503  0.007 440.681  895.362 beta=inf within 2
       2  1.259        1.500         inf 0.380 0.418 0.563  0.005 389.931  793.863 beta=inf within 2
       3  7.943        1.000       1.875 0.376 0.472 0.664  0.002 504.215 1022.430 beta=inf within 2
       4 15.849        1.500       2.250 0.370 0.413 0.550  0.007 430.313  874.626 beta=inf within 2
       5 19.953        1.875       1.500 0.377 0.416 0.566  0.011 463.137  940.274                  
       6 15.849        1.125       1.750 0.381 0.474 0.833  0.013 495.624 1005.247                  
       7 15.849        1.500       1.625 0.370 0.396 0.504  0.000 453.237  920.474                  
       8 15.849        1.000       1.125 0.132 0.388 0.712  0.000 418.191  850.382                  
       9 15.849        1.375       1.500 0.383 0.462 0.723  0.042 494.058 1002.116                  
      10 12.589        2.375       1.625 0.380 0.417 0.530  0.001 332.256  678.511                  
      11 15.849        1.625       2.250 0.373 0.404 0.653  0.012 416.345  846.690 beta=inf within 2
      12 15.849        1.875       1.750 0.378 0.405 0.488  0.008 444.361  902.722                  
      13 15.849        1.375       1.625 0.373 0.422 0.650  0.010 471.754  957.508                  
```

