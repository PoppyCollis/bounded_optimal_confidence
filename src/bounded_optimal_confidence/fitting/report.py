"""What a fit writes about itself: grid-edge flags, config_log.txt and report.md.

Comparisons between models (delta AIC, nested-model checks) are not here: see scripts/summarise_fits.py.
"""
from datetime import datetime

import numpy as np

from bounded_optimal_confidence.fitting.settings import (EDGE_TOL_NATS, LOG10_ALPHA_COARSE, LOG10_ALPHA_GRID,
                                                         LOG10_BETA_COARSE, LOG10_BETA_LO, SIG_X_GRID)


def edge_flags(best, profile, model):
    """Per subject, the grid edges (and nested values) reached by a search cell within EDGE_TOL_NATS of the subject's
    best: 'alpha lo', 'alpha inf within 2' (no decision noise does as well), 'sig_x=0', 'sig_x hi', and for the soft
    models 'beta lo', 'beta=inf within 2' (hard attention does as well) and 'beta~1 within 2' (the Bayes mixture
    does as well). A flag on a grid end means the optimum may lie beyond it."""
    out = []
    for r in best.itertuples():
        pr = profile[profile.subject == r.subject]
        ok = pr.profile_nll <= pr.profile_nll.min() + EDGE_TOL_NATS
        f = []
        fin = pr.log10_alpha[np.isfinite(pr.log10_alpha)]
        if (ok & (pr.log10_alpha == fin.min())).any():
            f.append("alpha lo")
        if (ok & np.isinf(pr.log10_alpha)).any():
            f.append("alpha inf within 2")
        if (ok & (pr.sig_x == 0)).any():
            f.append("sig_x=0")
        if (ok & (pr.sig_x == SIG_X_GRID[-1])).any():
            f.append("sig_x hi")
        if model.fits_beta:
            if (ok & np.isclose(pr.log10_beta, LOG10_BETA_LO)).any():
                f.append("beta lo")
            if (ok & np.isinf(pr.log10_beta)).any():
                f.append("beta=inf within 2")
            if (ok & np.isclose(pr.log10_beta, 0.0)).any():
                f.append("beta~1 within 2")
        out.append(",".join(f))
    return out


def write_config_log(path, model, subjects, commit, started, seed, n_reps, stride, n_refine):
    """config_log.txt: what was run, with which code and settings. `finished` is appended at the end."""
    rank_n, refit_n, clean_n = n_reps
    alphas = (f"{LOG10_ALPHA_COARSE.tolist()} (stage A), then +-0.5 at 0.125 around each subject's 3 best"
              if model.fits_beta else LOG10_ALPHA_GRID.tolist())
    betas = (f"{LOG10_BETA_COARSE.tolist()} (stage A), then +-0.25 at 0.125 within [{LOG10_BETA_LO}, 2.25], inf kept"
             if model.fits_beta else "-")
    lines = [f"started      {started:%Y-%m-%d %H:%M:%S}",
             f"git commit   {commit}",
             f"model        {model.name} ({model.label}), k {model.k}",
             f"data         exp 1, configs 1-4 pooled (one parameter set per subject), subjects {list(subjects)}",
             f"sig_x        {len(SIG_X_GRID)} values, 0 and 10^[-1, 2.5]"
             + (f"; stage A every {stride}th, stage B +-{stride} rungs" if model.fits_beta else ""),
             f"log10 alpha  {alphas}",
             f"log10 beta   {betas}",
             f"seed         {seed}",
             f"replicates   search {rank_n}, refit {refit_n}, clean {clean_n}; {n_refine} cells per subject refined"]
    path.write_text("\n".join(lines) + "\n")


def finish_config_log(path):
    with open(path, "a") as fh:
        fh.write(f"finished     {datetime.now():%Y-%m-%d %H:%M:%S}\n")


def beta_summary(best):
    """Report lines on the fitted log10 beta (inf = hard attention)."""
    n, lb = len(best), best.log10_beta
    fin = lb[np.isfinite(lb)]
    flag = lambda tag: int(best.edge.str.contains(tag, regex=False).sum())
    first = (f"- beta: inf (hard attention) for {int(np.isinf(lb).sum())}/{n}; finite fits median "
             f"**{10 ** fin.median():.3g}**, range [{10 ** fin.min():.3g}, {10 ** fin.max():.3g}]"
             if len(fin) else f"- beta: inf (hard attention) for {n}/{n}")
    return [first, f"- within {EDGE_TOL_NATS:g} nats of the best: beta = inf for {flag('beta=inf within 2')}/{n}, "
                   f"beta = 1 for {flag('beta~1 within 2')}/{n}, the grid bottom (beta {10 ** LOG10_BETA_LO:.3g}) "
                   f"for {flag('beta lo')}/{n}"]


def write_report(path, best, model, run_name):
    """report.md: the summed fit and the per-subject table. Returns the text."""
    cols = ["subject", "sig_x", "log10_alpha"] + (["log10_beta"] if model.fits_beta else []) \
        + ["b1", "b2", "b3", "lapse", "nll", "aic", "edge"]
    lines = [f"# {model.label} (`{model.name}`): exp 1, configs 1-4 pooled -- {run_name}", "",
             f"- summed NLL **{best.nll.sum():.1f}**, AIC **{best.aic.sum():.1f}** over {len(best)} subjects, "
             f"k = {model.k}"]
    if model.fits_beta:
        lines += beta_summary(best)
    lines += ["", "```", best[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"), "```", ""]
    text = "\n".join(lines)
    path.write_text(text + "\n")
    return text
