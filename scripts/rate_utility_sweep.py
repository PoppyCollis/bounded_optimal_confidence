"""Rate-utility curves in beta3: the 1-step rate-distortion frontier, the pinned symmetric top-2 hierarchy and the
solved bounded-optimal hierarchy, on every stimulus configuration, in the raw and the normalised world.

    python scripts/rate_utility_sweep.py                              # the paper's sweep: 640 cells, ~40 min on 14 workers
    python scripts/rate_utility_sweep.py --tag smoke --configs 1 3 --worlds raw --beta3-n 6   # shapes only, ~1 min
    python scripts/rate_utility_sweep.py --recheck <run dir>          # re-run the checks on a finished run

Writes results/rate_utility/rateutil_<date>_<time>[_<tag>]/{config_log.txt, curves.csv}. The paper's own run is
results/paper_rate_utility/rateutil_20260922_144534; paper_plots/fig_rate_cost_utility.py draws from it.
See docs/rate_utility.md.
"""
from bounded_optimal_confidence.bounded_rational.rate_utility import main

if __name__ == "__main__":
    main()
