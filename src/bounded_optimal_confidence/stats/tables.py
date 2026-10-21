"""The statistics as tables over a set of fits.

Every function takes `fits`, a {model name: (run folder, fit_results DataFrame)} as returned by `runs.load_fits`.
The per-subject log model evidence is approximated by -AIC / 2 throughout.
"""
from dataclasses import asdict

import pandas as pd

from bounded_optimal_confidence.models import MODELS
from bounded_optimal_confidence.stats.bms import bms, family_bms
from bounded_optimal_confidence.stats.paired import paired_test

REFERENCE = "diff"
# the family question of the paper: is attention (the Sym Top-2 models) or its absence (Diff) the more common?
TOP2_FAMILIES = {"No attention": ["diff"], "Sym Top-2": ["sym_top2_hard", "sym_top2_soft"]}


def aic_table(fits, models=None):
    """subjects x models DataFrame of AIC. Every run must cover the same subjects."""
    models = list(fits) if models is None else list(models)
    aic = pd.DataFrame({name: fits[name][1].set_index("subject").aic for name in models})
    if aic.isna().any().any():
        raise ValueError("the runs do not cover the same subjects")
    return aic


def log_evidence(aic):
    """K models x N subjects matrix of -AIC / 2 from a subjects x models AIC table."""
    return -0.5 * aic.to_numpy().T


def bms_table(fits, models=None):
    """Model-level BMS. Returns (one row per model: k, summed AIC, subjects best, frequency, XP, PXP; the BOR)."""
    aic = aic_table(fits, models)
    r = bms(log_evidence(aic))
    best = aic.idxmin(axis=1).value_counts()
    rows = [{"model": name, "label": MODELS[name].label, "k": int(fits[name][1].k.iloc[0]),
             "aic": aic[name].sum(), "best": int(best.get(name, 0)), "n": len(aic)} for name in aic]
    return pd.DataFrame(rows).assign(frequency=r.frequency, xp=r.xp, pxp=r.pxp), r.bor


def family_bms_tables(fits, families=TOP2_FAMILIES):
    """BMS restricted to the models of `families` ({family: [model names]}): at the model level, then at the family
    level under the Penny prior and under groupBMC's default prior. Returns [(title, table, BOR), ...]."""
    members = [m for ms in families.values() for m in ms]
    sizes = [len(ms) for ms in families.values()]
    aic = aic_table(fits, members)
    L = log_evidence(aic)
    best = aic.idxmin(axis=1).value_counts().reindex(members, fill_value=0)

    r = bms(L)
    model_level = pd.DataFrame({"model": members, "label": [MODELS[m].label for m in members],
                                "aic": aic.sum().to_numpy(), "best": best.to_numpy(),
                                "frequency": r.frequency, "xp": r.xp, "pxp": r.pxp})
    out = [("model level (prior 1 per model)", model_level, r.bor)]
    best_family = [int(best[ms].sum()) for ms in families.values()]
    for prior, title in (("penny", "family level (prior 1 per family, split equally inside: Penny et al. 2010)"),
                         ("groupbmc", "family level (groupBMC's default prior: counts sum to 1)")):
        r = family_bms(L, sizes, prior)
        out.append((title, pd.DataFrame({"family": list(families), "best": best_family,
                                         "frequency": r.frequency, "xp": r.xp, "pxp": r.pxp}), r.bor))
    return out


def paired_table(fits, reference=REFERENCE, models=None):
    """One row per model other than the reference: `paired.paired_test` of its AIC minus the reference's."""
    aic = aic_table(fits, models)
    rows = [{"model": name, "label": MODELS[name].label, **asdict(paired_test(aic[name], aic[reference]))}
            for name in aic if name != reference]
    return pd.DataFrame(rows)
