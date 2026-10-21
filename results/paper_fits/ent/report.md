# entropy_alpha_sigx_pooled, exp 1, cfgs [1, 2, 3, 4] pooled -- entropy_alpha_sigx_pooled_cfg1234_20261004_111153

## fit

- summed NLL **6748.5**, AIC **13653.0** over 13 subjects, k = 6
- no per-config run for every cfg; pooling cost not computed
- vs pooled top2hard (same k; AICs only):
- vs k = 6 `entropy_top2_hard_alpha_sigx_pooled_cfg1234_20260926_193133` on 13 subjects: summed delta NLL **+950.1**, delta AIC **+1900.3** (negative favours this model); entropy wins on AIC for 0/13
- vs pooled diff (same k, same posterior; only the readout differs):
- vs k = 6 `diff_alpha_sigx_pooled_cfg1234_20260928_151314` on 13 subjects: summed delta NLL **+968.5**, delta AIC **+1937.0** (negative favours this model); entropy wins on AIC for 0/13


```
 subject  sig_x  log10_alpha    b1    b2    b3  lapse     nll      aic    edge
       1  0.158        0.750 0.171 0.394 0.603  0.041 548.872 1109.743        
       2  0.158        0.500 0.383 0.497 0.808  0.000 533.427 1078.855 sig_x=0
       3  0.200        0.000 0.413 0.844 0.996  0.000 554.851 1121.702 sig_x=0
       4  0.501        0.750 0.092 0.411 0.654  0.021 496.835 1005.670 sig_x=0
       5  0.100        0.500 0.386 0.536 0.837  0.000 539.229 1090.458        
       6  0.100        0.500 0.377 0.580 0.976  0.010 519.665 1051.331        
       7  0.631        0.750 0.130 0.390 0.596  0.000 489.745  991.491 sig_x=0
       8  0.158        0.500 0.029 0.390 0.874  0.000 444.486  900.972 sig_x=0
       9  0.398        0.500 0.389 0.582 0.925  0.000 538.815 1089.631        
      10  0.398        0.750 0.385 0.490 0.681  0.000 500.590 1013.180 sig_x=0
      11  7.943        0.750 0.210 0.385 0.773  0.000 503.249 1018.497 sig_x=0
      12  0.316        0.500 0.388 0.483 0.709  0.000 557.647 1127.294        
      13 12.589        1.000 0.184 0.384 0.662  0.010 521.074 1054.149 sig_x=0
```

