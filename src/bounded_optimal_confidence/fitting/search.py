"""The search stage: score grid cells, with the b-triple and lapse profiled out at each.

    full_grid_search        the k = 6 models: every sig_x x every log10 alpha (37 x 20 cells per subject)
    coarse_to_fine_search   the soft-attention models (k = 7): the full product with beta would be ~20,000 cells per
                            subject, so stage A scores a coarse grid and stage B refines around each subject's best

Both return one row per (subject, cell): sig_x_index, sig_x, log10_alpha, [log10_beta], profile_nll, b1-b3, lapse.
"""
import pandas as pd

from bounded_optimal_confidence.fitting.settings import (LOG10_ALPHA_COARSE, LOG10_ALPHA_GRID, LOG10_BETA_COARSE,
                                                         N_SEED, SIG_X_GRID, fine_log10_alphas, fine_log10_betas)
from bounded_optimal_confidence.fitting.tasks import SearchTask, score_cells
from bounded_optimal_confidence.utils.parallel import run_parallel


def _score(tasks, n_workers, label):
    return pd.DataFrame([row for rows in run_parallel(score_cells, tasks, n_workers, label, every=20) for row in rows])


def full_grid_search(model, subjects, n_reps, seed, n_workers):
    """Every (sig_x, log10 alpha) cell; one task per (subject, sig_x)."""
    cells = tuple((float(la), None) for la in LOG10_ALPHA_GRID)
    tasks = [SearchTask(model.name, s, j, cells, n_reps, seed) for s in subjects for j in range(len(SIG_X_GRID))]
    print(f"search: {len(SIG_X_GRID)} sig_x x {len(cells)} alphas x {len(subjects)} subjects", flush=True)
    return _score(tasks, n_workers, "search")


def coarse_to_fine_search(model, subjects, n_reps, seed, n_workers, stride):
    """Stage A: every `stride`-th sig_x x LOG10_ALPHA_COARSE x LOG10_BETA_COARSE. Stage B: for each of a subject's
    N_SEED best stage-A cells, +-stride sig_x rungs x the fine alphas x the fine betas around it (cells already scored
    in stage A are not repeated). Returns both stages, marked in a `stage` column."""
    n_sig_x = len(SIG_X_GRID)
    coarse = tuple((float(la), float(lb)) for la in LOG10_ALPHA_COARSE for lb in LOG10_BETA_COARSE)
    coarse_j = list(range(0, n_sig_x, stride))
    print(f"stage A: {len(coarse_j)} sig_x x {len(LOG10_ALPHA_COARSE)} alphas x {len(LOG10_BETA_COARSE)} betas x "
          f"{len(subjects)} subjects", flush=True)
    a = _score([SearchTask(model.name, s, j, coarse, n_reps, seed) for j in coarse_j for s in subjects],
               n_workers, "stage A")
    a["stage"] = "A"

    jobs = {}                                       # (sig_x_index, subject) -> cells to score
    for s, g in a.groupby("subject"):
        for r in g.sort_values("profile_nll").head(N_SEED).itertuples():
            cells = {(la, float(lb)) for la in fine_log10_alphas(r.log10_alpha) for lb in fine_log10_betas(r.log10_beta)}
            for jj in range(max(r.sig_x_index - stride, 0), min(r.sig_x_index + stride + 1, n_sig_x)):
                jobs.setdefault((jj, int(s)), set()).update(cells)
    done = set(zip(a.subject, a.sig_x_index, a.log10_alpha, a.log10_beta))
    tasks = []
    for (jj, s), cells in sorted(jobs.items()):
        todo = sorted(c for c in cells if (s, jj, c[0], c[1]) not in done)
        if todo:
            tasks.append(SearchTask(model.name, s, jj, tuple(todo), n_reps, seed))
    print(f"stage B: {sum(len(t.cells) for t in tasks)} (subject, sig_x, alpha, beta) cells", flush=True)
    b = _score(tasks, n_workers, "stage B")
    b["stage"] = "B"
    return pd.concat([a, b], ignore_index=True)


def search(model, subjects, n_reps, seed, n_workers, stride):
    """The search for this model: coarse-to-fine if it fits beta, the full grid otherwise."""
    if model.fits_beta:
        return coarse_to_fine_search(model, subjects, n_reps, seed, n_workers, stride)
    return full_grid_search(model, subjects, n_reps, seed, n_workers)
