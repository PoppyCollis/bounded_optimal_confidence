"""Paired tests and the bootstrap: scipy agreement, the sign convention, and the paper's numbers."""
import numpy as np
import pytest
from scipy import stats

from bounded_optimal_confidence.runs import load_fits
from bounded_optimal_confidence.stats import bootstrap_ci, paired_test
from bounded_optimal_confidence.stats.tables import aic_table, paired_table

FITS = load_fits(paper=True)
AIC = aic_table(FITS)


def test_matches_scipy_and_the_sign_convention():
    rng = np.random.default_rng(0)
    ref = rng.normal(800, 50, 13)
    model = ref + rng.normal(5, 8, 13)                          # the model is worse: positive differences
    r = paired_test(model, ref)
    d = model - ref
    t = stats.ttest_rel(model, ref)
    assert (r.t, r.p, r.df) == (t.statistic, t.pvalue, 12)
    assert r.sum == pytest.approx(d.sum()) and r.mean == pytest.approx(d.mean())
    assert r.sd == pytest.approx(d.std(ddof=1)) and r.d_z == pytest.approx(d.mean() / d.std(ddof=1))
    assert r.wilcoxon_p == stats.wilcoxon(d).pvalue
    assert r.n_better == (d < 0).sum()
    assert r.mean > 0 and r.t > 0
    assert r.ci_lo < r.mean < r.ci_hi and r.sum_ci_lo < r.sum < r.sum_ci_hi


def test_bootstrap_ci_brackets_the_statistic_and_is_reproducible():
    x = np.arange(13, dtype=float)
    lo, hi = bootstrap_ci(x, stat=np.sum)
    assert lo < x.sum() < hi
    assert (lo, hi) == bootstrap_ci(x, stat=np.sum)
    assert bootstrap_ci(np.full(5, 2.0)) == (2.0, 2.0)
    with pytest.raises(ValueError):
        bootstrap_ci(np.ones((2, 2)))


def test_paper_top2_models_do_not_differ_from_diff():
    tab = paired_table(FITS).set_index("model")
    hard, soft = tab.loc["sym_top2_hard"], tab.loc["sym_top2_soft"]
    assert hard["mean"] == pytest.approx(2.83, abs=0.01) and hard.sd == pytest.approx(8.56, abs=0.01)
    assert hard.t == pytest.approx(1.19, abs=0.01) and hard.p == pytest.approx(0.257, abs=0.001)
    assert (hard.ci_lo, hard.ci_hi) == pytest.approx((-2.34, 8.00), abs=0.01)
    assert hard.d_z == pytest.approx(0.33, abs=0.01) and hard.wilcoxon_p == pytest.approx(0.1465, abs=1e-4)
    assert hard.n_better == 4
    assert soft["mean"] == pytest.approx(-1.98, abs=0.01) and soft.t == pytest.approx(-0.90, abs=0.01)
    assert soft.p == pytest.approx(0.387, abs=0.001) and soft.wilcoxon_p == pytest.approx(0.635, abs=0.001)
    assert soft.n_better == 6


def test_paper_summed_delta_aic_and_bootstrap_intervals():
    tab = paired_table(FITS).set_index("model")
    assert tab["sum"]["ent"] == pytest.approx(1937.0, abs=0.1)
    assert (tab.sum_ci_lo["ent"], tab.sum_ci_hi["ent"]) == pytest.approx((1353.4, 2550.3), abs=0.1)
    assert (tab.sum_ci_lo["sym_top2_hard"], tab.sum_ci_hi["sym_top2_hard"]) == pytest.approx((-26.2, 89.5), abs=0.1)
    assert (tab.sum_ci_lo["sym_top2_soft"], tab.sum_ci_hi["sym_top2_soft"]) == pytest.approx((-79.8, 25.6), abs=0.1)
    others = tab.drop(["sym_top2_hard", "sym_top2_soft"])
    assert (others.sum_ci_lo > 0).all() and (others.p < 1e-3).all()


def test_paired_table_covers_every_model_but_the_reference():
    tab = paired_table(FITS)
    assert tab.model.tolist() == [m for m in FITS if m != "diff"]
    assert (tab.n == 13).all() and (tab.df == 12).all()
