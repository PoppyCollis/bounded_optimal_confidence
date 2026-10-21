# entropy_ci_asym_right_alpha_sigx_pooled, exp 1, cfgs [1, 2, 3, 4] pooled -- entropy_ci_asym_right_alpha_sigx_pooled_cfg1234_20261005_233324

## fit

- summed NLL **8681.2**, AIC **17518.3** over 13 subjects, k = 6
- no per-config run for every cfg; pooling cost not computed
- vs pooled isi_asym_right (same masks and k; cuts at the midpoints instead of model-derived):
- vs k = 6 `entropy_isi_asym_right_alpha_sigx_pooled_cfg1234_20261003_194452` on 13 subjects: summed delta NLL **+168.8**, delta AIC **+337.7** (negative favours this model); ci_asym_right wins on AIC for 1/13
- vs pooled top2hard (same k; AICs only):
- vs k = 6 `entropy_top2_hard_alpha_sigx_pooled_cfg1234_20260926_193133` on 13 subjects: summed delta NLL **+2882.8**, delta AIC **+5765.7** (negative favours this model); ci_asym_right wins on AIC for 0/13
- vs pooled diff (same k; AICs only):
- vs k = 6 `diff_alpha_sigx_pooled_cfg1234_20260928_151314` on 13 subjects: summed delta NLL **+2901.2**, delta AIC **+5802.4** (negative favours this model); ci_asym_right wins on AIC for 0/13


```
 subject  sig_x  log10_alpha    b1    b2    b3  lapse     nll      aic               edge
       1 19.953        3.250 0.007 0.072 0.250  0.460 694.161 1400.322 alpha inf within 2
       2  0.100        2.000 0.012 0.066 0.366  0.463 648.731 1309.461            sig_x=0
       3 25.119        2.000 0.005 0.170 1.000  0.522 725.965 1463.929 alpha inf within 2
       4 15.849        1.500 0.000 0.133 0.454  0.450 682.006 1376.012                   
       5 25.119        2.000 0.041 0.193 1.000  0.504 716.238 1444.476 alpha inf within 2
       6 19.953        1.500 0.018 0.199 1.000  0.456 687.781 1387.562                   
       7 25.119        1.500 0.003 0.106 0.428  0.382 676.702 1365.403                   
       8 15.849        1.500 0.000 0.044 1.000  0.293 553.719 1119.438                   
       9 15.849        1.500 0.033 0.208 1.000  0.485 698.565 1409.129 alpha inf within 2
      10 25.119        2.500 0.054 0.181 0.391  0.323 642.428 1296.855 alpha inf within 2
      11 12.589        2.500 0.007 0.055 1.000  0.403 610.143 1232.287 alpha inf within 2
      12 19.953        2.000 0.024 0.107 0.287  0.404 666.652 1345.304                   
      13 12.589        1.000 0.020 0.221 1.000  0.410 678.085 1368.170                   
```

