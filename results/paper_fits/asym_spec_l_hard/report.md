# entropy_ci_asym_left_alpha_sigx_pooled, exp 1, cfgs [1, 2, 3, 4] pooled -- entropy_ci_asym_left_alpha_sigx_pooled_cfg1234_20261005_224936

## fit

- summed NLL **8842.8**, AIC **17841.6** over 13 subjects, k = 6
- no per-config run for every cfg; pooling cost not computed
- vs pooled isi_asym_left (same masks and k; cuts at the midpoints instead of model-derived):
- vs k = 6 `entropy_isi_asym_left_alpha_sigx_pooled_cfg1234_20261003_191021` on 13 subjects: summed delta NLL **+144.0**, delta AIC **+287.9** (negative favours this model); ci_asym_left wins on AIC for 0/13
- vs pooled top2hard (same k; AICs only):
- vs k = 6 `entropy_top2_hard_alpha_sigx_pooled_cfg1234_20260926_193133` on 13 subjects: summed delta NLL **+3044.5**, delta AIC **+6088.9** (negative favours this model); ci_asym_left wins on AIC for 0/13
- vs pooled diff (same k; AICs only):
- vs k = 6 `diff_alpha_sigx_pooled_cfg1234_20260928_151314` on 13 subjects: summed delta NLL **+3062.8**, delta AIC **+6125.7** (negative favours this model); ci_asym_left wins on AIC for 0/13


```
 subject  sig_x  log10_alpha    b1    b2    b3  lapse     nll      aic               edge
       1 15.849        1.750 0.001 0.069 0.333  0.403 653.960 1319.919                   
       2  6.310        2.000 0.016 0.059 0.293  0.515 695.433 1402.866                   
       3 19.953        1.750 0.008 1.000 1.000  0.626 742.578 1497.156 alpha inf within 2
       4 19.953        1.750 0.000 0.071 0.367  0.405 669.572 1351.143                   
       5 25.119        1.750 0.027 0.169 0.539  0.425 708.821 1429.643                   
       6 10.000        1.250 0.018 0.178 1.000  0.445 677.248 1366.496                   
       7 25.119        2.000 0.000 0.069 0.294  0.447 691.052 1394.104 alpha inf within 2
       8 19.953        1.000 0.000 0.044 1.000  0.350 617.463 1246.926                   
       9 15.849        1.250 0.043 0.247 1.000  0.504 714.455 1440.910                   
      10 12.589        2.250 0.026 0.101 0.306  0.437 661.023 1334.045                   
      11 15.849        1.500 0.008 0.094 1.000  0.390 625.570 1263.139                   
      12 25.119        2.500 0.018 0.070 0.224  0.434 694.848 1401.696 alpha inf within 2
      13 12.589        1.500 0.004 0.110 1.000  0.499 690.776 1393.551            sig_x=0
```

