"""Re-solve cells of the paper's rate-utility sweep with this package and confirm they are reproduced exactly.

    python scripts/check_paper_rate_utility.py                 # 8 cells spread over both worlds, the 4 configs and the beta3 axis (~2 min)
    python scripts/check_paper_rate_utility.py --n 32          # 32 cells evenly spaced along the stored run
    python scripts/check_paper_rate_utility.py --all           # all 640 cells (the full sweep, ~40 min on 14 workers)

Each cell gives three rows (top2, solved, rd). Every column of every row is compared with the stored curves.csv after
the same CSV round trip the sweep writes with, so "reproduced" means the file would have come out identical. Exits
non-zero on any difference.
"""
import argparse
from concurrent.futures import as_completed

import pandas as pd

from bounded_optimal_confidence.bounded_rational import paper_run
from bounded_optimal_confidence.bounded_rational.rate_utility import BETA2, cell, world_kw_for
from bounded_optimal_confidence.bounded_rational.settings import N_WORKERS
from bounded_optimal_confidence.utils.parallel import worker_pool

# The default spot check: the bottom rung, the beta3 = 1 mark and the top rung of the axis, spread over both worlds
# and all four configurations, with two interior rungs.
DEFAULT_CELLS = [("raw", 1, 0), ("raw", 2, "mark"), ("raw", 3, -1), ("raw", 4, 40),
                 ("norm", 1, -1), ("norm", 2, 40), ("norm", 3, "mark"), ("norm", 4, 0)]


def stored_cells(curves):
    """Every (world, config, beta3) of the stored run, in the stored order."""
    keys = curves[["world", "stimulus_config", "beta3"]].drop_duplicates()
    return [(w, int(c), float(b)) for w, c, b in keys.itertuples(index=False)]


def pick(curves, args):
    cells = stored_cells(curves)
    if args.all:
        return cells
    if args.n:
        step = max(1, len(cells) // args.n)
        return cells[::step][:args.n]
    axis = sorted(set(b for _w, _c, b in cells))
    out = []
    for w, c, i in DEFAULT_CELLS:
        b = 1.0 if i == "mark" else axis[i]
        out.append((w, c, b))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--n", type=int, default=0, help="number of cells, evenly spaced along the stored run")
    p.add_argument("--all", action="store_true", help="every stored cell")
    p.add_argument("--workers", type=int, default=N_WORKERS)
    args = p.parse_args()

    curves = paper_run.load_paper_curves()
    q, beta1 = float(curves["q"].iloc[0]), float(curves["beta1"].iloc[0])
    cells = pick(curves, args)
    tasks = [(world_kw_for(w), c, b, q, (beta1, BETA2), True) for w, c, b in cells]
    print(f"{len(tasks)} cells of {paper_run.PAPER_RUN_DIR.name}, {min(args.workers, len(tasks))} workers")

    rows = []
    with worker_pool(min(args.workers, len(tasks))) as ex:
        futures = [ex.submit(cell, t) for t in tasks]
        for n, fut in enumerate(as_completed(futures), 1):
            rows += fut.result()
            print(f"  {n}/{len(tasks)}", flush=True)

    new = paper_run.as_written(rows, list(curves.columns))
    new = new.sort_values(paper_run.KEY).reset_index(drop=True)
    result = paper_run.compare(new, curves)
    per_row = result.drop(columns=paper_run.KEY).all(axis=1)
    table = result[paper_run.KEY].assign(reproduced=per_row.to_numpy())
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(table.to_string(index=False))
    if per_row.all():
        print(f"\nall {len(new)} rows of {len(cells)} cells reproduced exactly")
        return 0
    bad = result.drop(columns=paper_run.KEY).columns[~result.drop(columns=paper_run.KEY).all(axis=0)]
    print(f"\nFAILED: {int((~per_row).sum())} rows differ; columns {list(bad)}")
    print("largest differences:", paper_run.worst_differences(new, curves))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
