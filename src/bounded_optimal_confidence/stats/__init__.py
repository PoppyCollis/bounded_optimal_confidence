"""Group-level statistics on the fits.

    bms         random-effects Bayesian model selection (frequencies, exceedance probabilities, omnibus risk)
    paired      paired t-test, Wilcoxon test and bootstrap of one model against another on per-subject AIC
    bootstrap   percentile bootstrap confidence interval
    tables      the above as tables over a set of fits
"""
from bounded_optimal_confidence.stats.bms import BMSResult, bms, family_bms
from bounded_optimal_confidence.stats.bootstrap import bootstrap_ci
from bounded_optimal_confidence.stats.paired import PairedTest, paired_test

__all__ = ["BMSResult", "bms", "family_bms", "bootstrap_ci", "PairedTest", "paired_test"]
