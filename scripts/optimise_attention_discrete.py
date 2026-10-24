"""The shared attention optimum over sampled worlds: sparse structure priors p(x|phi) and the optimal hard router
p(phi|w), with w discrete on a grid, for one box size B or swept over B.

    python scripts/optimise_attention_discrete.py --world-prior recentred --B 250           # one B: figures of the optimum
    python scripts/optimise_attention_discrete.py --world-prior recentred \\
        --sweep-B 64 125 250 500 1000 2000 --n-seeds 5 --n-workers 2                        # the paper's sweep, ~25 min

Outputs go under results/shared_attention/. The B sweep is memory-bound (a B = 2000 job peaks at 2.3 GB), so
--n-workers 2 is what the paper's run used; the script's own default is 4. The paper's sweep is
results/paper_shared_attention/optimise_attention_discrete_Bsweep_recentred_n10000_sig64_20261012_150128;
scripts/check_paper_shared_attention.py reproduces it and paper_plots/fig_pxgphi_vs_B.py draws from it.
See docs/shared_attention.md.
"""
from bounded_optimal_confidence.shared_attention.optimise_attention_discrete import main

if __name__ == "__main__":
    main()
