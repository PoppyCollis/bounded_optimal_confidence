"""Re-solve (B, seed) jobs of the paper's shared-attention B sweep and confirm they are reproduced exactly.

    python scripts/check_paper_shared_attention.py                      # all 30 jobs on 2 workers (~25 min, memory-bound)
    python scripts/check_paper_shared_attention.py --jobs 64:0 250:0    # a subset, as B:seed

Every column of sweep.csv but the two timing ones (seconds, peak_gb) is compared with the stored row after the same
CSV round trip the sweep writes with. Exits non-zero on any difference.
"""
import argparse
import multiprocessing

import pandas as pd

from bounded_optimal_confidence.shared_attention import paper_run
from bounded_optimal_confidence.shared_attention.optimise_attention_discrete import _sweep_job


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--jobs", nargs="+", default=None, metavar="B:SEED", help="a subset of the stored jobs")
    p.add_argument("--workers", type=int, default=2, help="parallel jobs; each holds its worlds on the grid (up to 2.3 GB)")
    args = p.parse_args()

    stored = paper_run.load_paper_sweep()
    keys = [(float(b), int(s)) for b, s in stored[paper_run.KEY].itertuples(index=False)]
    if args.jobs:
        want = [(float(j.split(":")[0]), int(j.split(":")[1])) for j in args.jobs]
        missing = [k for k in want if k not in keys]
        if missing:
            raise SystemExit(f"not in the stored sweep: {missing}")
        keys = want
    # largest B first, as run_sweep orders them, so the long jobs do not trail at the end
    keys.sort(key=lambda k: (-k[0], k[1]))
    jobs = [(paper_run.PAPER_CFG, b, s) for b, s in keys]
    print(f"{len(jobs)} jobs of {paper_run.PAPER_RUN_DIR.name} on {args.workers} workers")

    rows = []
    with multiprocessing.Pool(args.workers, maxtasksperchild=1) as pool:
        for row in pool.imap_unordered(_sweep_job, jobs):
            rows.append(row)
            print(f"  [{len(rows):3d}/{len(jobs)}] B {row['B']:g} seed {row['seed']}: {row['supports']}, "
                  f"gap {row['gap_opt_held']:.4f}, {row['seconds']:.0f} s, {row['peak_gb']:.2f} GB", flush=True)
    rows.sort(key=lambda r: (r["B"], r["seed"]))

    new = paper_run.as_written(rows)
    result = paper_run.compare(new, stored)
    per_row = result.drop(columns=paper_run.KEY).all(axis=1)
    table = result[paper_run.KEY].assign(reproduced=per_row.to_numpy())
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(table.to_string(index=False))
    merged = new.merge(stored, on=paper_run.KEY, suffixes=("", "_stored"))
    print(f"\n(timing, not compared) seconds: new {new['seconds'].sum():.0f} s in all, "
          f"stored {merged['seconds_stored'].sum():.0f} s")
    if per_row.all():
        print(f"all {len(new)} rows reproduced exactly")
        return 0
    bad = result.drop(columns=paper_run.KEY).columns[~result.drop(columns=paper_run.KEY).all(axis=0)]
    print(f"\nFAILED: {int((~per_row).sum())} rows differ; columns {list(bad)}")
    print("largest differences:", paper_run.worst_differences(new, stored))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
