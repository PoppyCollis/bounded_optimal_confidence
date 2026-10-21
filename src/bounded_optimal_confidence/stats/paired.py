"""Paired comparison of two models on their per-subject AIC.

The difference d = AIC(model) - AIC(reference) is taken subject by subject, so a positive d means the reference
fits that subject better. The summed d is the delta AIC of the model tables; the tests ask whether its sign holds
across subjects: a two-tailed paired t-test on d, the exact Wilcoxon signed-rank test as a check that does not
assume normality, and a percentile bootstrap of the sum (`bootstrap.bootstrap_ci`).
"""
from dataclasses import dataclass

import numpy as np
from scipy import stats

from bounded_optimal_confidence.stats.bootstrap import bootstrap_ci


@dataclass(frozen=True)
class PairedTest:
    n: int              # subjects
    sum: float          # summed difference: the model's delta AIC against the reference
    mean: float         # mean difference per subject
    sd: float           # standard deviation of the differences (n - 1)
    t: float            # paired t statistic
    df: int             # n - 1
    p: float            # two-tailed p of the t-test
    ci_lo: float        # 95% confidence interval of the mean difference
    ci_hi: float
    d_z: float          # effect size: mean / sd
    wilcoxon_p: float   # two-tailed exact Wilcoxon signed-rank test
    n_better: int       # subjects the model fits better than the reference (d < 0)
    sum_ci_lo: float    # bootstrap 95% confidence interval of the sum
    sum_ci_hi: float


def paired_test(model_aic, ref_aic):
    """The paired tests of model minus reference, given the two per-subject AIC vectors in the same subject order."""
    model_aic, ref_aic = np.asarray(model_aic, dtype=float), np.asarray(ref_aic, dtype=float)
    if model_aic.shape != ref_aic.shape or model_aic.ndim != 1:
        raise ValueError("the two AIC vectors must be 1-D and of the same length")
    d = model_aic - ref_aic
    t = stats.ttest_rel(model_aic, ref_aic)
    ci = t.confidence_interval()
    sd = d.std(ddof=1)
    sum_lo, sum_hi = bootstrap_ci(d, stat=np.sum)
    return PairedTest(n=len(d), sum=float(d.sum()), mean=float(d.mean()), sd=float(sd), t=float(t.statistic),
                      df=int(t.df), p=float(t.pvalue), ci_lo=float(ci.low), ci_hi=float(ci.high),
                      d_z=float(d.mean() / sd), wilcoxon_p=float(stats.wilcoxon(d).pvalue),
                      n_better=int((d < 0).sum()), sum_ci_lo=sum_lo, sum_ci_hi=sum_hi)
