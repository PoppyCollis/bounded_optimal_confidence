# entropy_ci_sym_alpha_sigx_pooled, exp 1, cfgs [1, 2, 3, 4] pooled -- entropy_ci_sym_alpha_sigx_pooled_cfg1234_20261006_001634

## fit

- summed NLL **10049.4**, AIC **20254.7** over 13 subjects, k = 6
- no per-config run for every cfg; pooling cost not computed
- vs pooled isi_sym (same masks and k; cuts at the midpoints instead of model-derived):
- vs k = 6 `entropy_isi_sym_alpha_sigx_pooled_cfg1234_20261003_183613` on 13 subjects: summed delta NLL **+469.4**, delta AIC **+938.7** (negative favours this model); ci_sym wins on AIC for 0/13
- vs pooled top2hard (same k; AICs only):
- vs k = 6 `entropy_top2_hard_alpha_sigx_pooled_cfg1234_20260926_193133` on 13 subjects: summed delta NLL **+4251.0**, delta AIC **+8502.0** (negative favours this model); ci_sym wins on AIC for 0/13
- vs pooled diff (same k; AICs only):
- vs k = 6 `diff_alpha_sigx_pooled_cfg1234_20260928_151314` on 13 subjects: summed delta NLL **+4269.4**, delta AIC **+8538.8** (negative favours this model); ci_sym wins on AIC for 0/13


```
 subject  sig_x  log10_alpha    b1    b2    b3  lapse     nll      aic                                edge
       1  0.100          inf 0.000 0.412 1.000  0.825 795.781 1603.562          alpha inf within 2,sig_x=0
       2  6.310          inf 0.000 0.376 1.000  0.754 771.605 1555.210                  alpha inf within 2
       3 31.623        2.000 0.000 1.000 1.000  0.750 781.651 1575.303                  alpha inf within 2
       4 31.623        3.250 0.000 1.000 1.000  0.742 774.964 1561.927         alpha lo,alpha inf within 2
       5  1.585        2.000 0.000 0.302 1.000  0.841 803.370 1618.741 alpha lo,alpha inf within 2,sig_x=0
       6 10.000        1.250 0.313 0.724 1.000  0.713 768.437 1548.875                            alpha lo
       7  0.100          inf 0.396 0.559 1.000  0.813 791.503 1595.006 alpha lo,alpha inf within 2,sig_x=0
       8 15.849        4.000 0.000 0.324 1.000  0.604 697.389 1406.779                  alpha inf within 2
       9  7.943        1.000 0.000 0.519 1.000  0.805 795.825 1603.650         alpha lo,alpha inf within 2
      10  0.000          inf 0.000 0.353 0.559  0.844 803.089 1618.178          alpha inf within 2,sig_x=0
      11 12.589        3.750 0.196 0.441 1.000  0.597 696.742 1405.485                  alpha inf within 2
      12  5.012        4.000 0.000 0.274 0.458  0.802 790.329 1592.658          alpha inf within 2,sig_x=0
      13  0.398        1.750 0.000 0.281 1.000  0.780 778.670 1569.340          alpha inf within 2,sig_x=0
```

