"""The paper's fits (results/paper_fits): complete, consistent, and reproduced by this package's code."""
import numpy as np
import pytest

from bounded_optimal_confidence.experiment import SUBJECTS
from bounded_optimal_confidence.fitting.settings import CLEAN_SIMNTRL_FACTOR, SEED, SIG_X_GRID, SIMNTRL
from bounded_optimal_confidence.fitting.tasks import CleanTask, clean_score
from bounded_optimal_confidence.models import MODELS
from bounded_optimal_confidence.runs import load_fits

FITS = load_fits(paper=True)


def test_every_model_has_a_complete_fit():
    assert list(FITS) == list(MODELS)
    for name, (_run, fit) in FITS.items():
        assert fit.subject.tolist() == list(SUBJECTS)
        assert (fit.k == MODELS[name].k).all()
        assert np.allclose(fit.aic, 2 * (fit.k + fit.nll))
        assert np.array_equal(fit.sig_x, SIG_X_GRID[fit.sig_x_index])
        assert ("log10_beta" in fit) == MODELS[name].fits_beta


@pytest.mark.slow
@pytest.mark.parametrize("name", ["diff", "sym_top2_hard", "sym_spec_soft"])
def test_rescore_reproduces_the_paper(name):
    """Subject 1's best cell, re-scored: bit-identical to the stored clean NLL, b-triple and lapse."""
    r = FITS[name][1].iloc[0]
    row = clean_score(CleanTask(name, int(r.subject), int(r.sig_x_index), float(r.log10_alpha),
                                float(r.log10_beta) if MODELS[name].fits_beta else None, SIMNTRL,
                                CLEAN_SIMNTRL_FACTOR * SIMNTRL, SEED))
    for c in ("nll", "refit_profile_nll", "b1", "b2", "b3", "lapse"):
        assert row[c] == r[c], (c, row[c], r[c])
