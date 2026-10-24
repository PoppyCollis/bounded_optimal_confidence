"""The 1D Bayesian illustration of the confidence readouts, for the paper's explanatory figures.

K options x with a flat prior, Gaussian likelihoods p(w|x) = N(w; mu_x, sigma); the observation w is swept over a
grid and each readout of the posterior p(x|w) is traced:
    Entropy model:       -H[p(x|w)]
    Max model:           max_x p(x|w)
    Difference model:    top-1 minus top-2 posterior probability
    Entropy Top2 model:  -H of the posterior kept to its two most likely options and renormalised
each rescaled to [0, 1], either over the sweep (norm="data") or by the lowest and highest values the readout can take
for K options (norm="theory": 0 = flat posterior, 1 = one option certain). With K = 2 the Max and Difference models
coincide exactly.

These are the illustrations only: the readouts the fitted models use (log2-normalised, with the Dirichlet decision
noise) are in `models/readouts.py`.

From `ITBR_confidence/paper_plots/bayes_1d_readouts.py` (its own one-axis figure dropped) and `likelihoods` from
`paper_plots/fig_likelihood_confidence.py`. `style_axis` and `draw_traces` set this module's own text sizes and
READOUT_STYLE colours; `paper_plots/fig_likelihood_confidence.py` restyles the axes afterwards.
"""
import argparse

import numpy as np

MU = (-1.0, 1.0)
SIGMA = 1.0
PRIOR = (0.5, 0.5)
W = np.linspace(-4, 4, 801)

TICK_SIZE = 16
LABEL_SIZE = 19
LEGEND_SIZE = 15

# Okabe-Ito colours: blue, orange, black, sky blue are pairwise >= 18.9 OKLab units apart under
# simulated protan/deutan/tritan vision; distinct line styles too, so identity is not colour-alone.
# Shared by every figure so a readout looks the same everywhere.
READOUT_STYLE = {
    "Entropy model": ("#0072B2", "-"),
    "Max model": ("#E69F00", "-"),
    "Difference model": ("#000000", (0, (3, 3))),
    "Entropy Top2 model": ("#56B4E9", "-."),
}


def likelihoods(w, mu, sigma):
    """Normalised Gaussian densities p(y|x), shape (len(w), K)."""
    return np.exp(-0.5 * ((w[:, None] - np.array(mu)) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def posterior(w, mu=MU, sigma=SIGMA, prior=PRIOR):
    """p(x|w), shape (len(w), K)."""
    # log space, shifted by the row max, so far-tail w doesn't underflow to 0/0
    log_joint = -0.5 * ((w[:, None] - np.array(mu)) / sigma) ** 2 + np.log(prior)
    joint = np.exp(log_joint - log_joint.max(axis=1, keepdims=True))
    return joint / joint.sum(axis=1, keepdims=True)


def neg_entropy(post):
    """-H[p(x|w)] in nats."""
    return np.sum(post * np.log(np.clip(post, 1e-300, None)), axis=1)


def max_prob(post):
    """max_x p(x|w)."""
    return post.max(axis=1)


def prob_difference(post):
    """Top-1 minus top-2 posterior probability."""
    top2 = np.sort(post, axis=1)[:, -2:]
    return top2[:, 1] - top2[:, 0]


def top2_neg_entropy(post):
    """Keep the two most likely options, renormalise, then -H."""
    top2 = np.sort(post, axis=1)[:, -2:]
    return neg_entropy(top2 / top2.sum(axis=1, keepdims=True))


def rescale(y, bounds, norm):
    """Map a confidence trace to [0, 1]: by its own min/max over the sweep (norm="data"), or by the
    readout's theoretical (lowest, highest) values (norm="theory")."""
    lo, hi = (y.min(), y.max()) if norm == "data" else bounds
    return (y - lo) / (hi - lo)


def parse_norm():
    parser = argparse.ArgumentParser()
    parser.add_argument("--norm", choices=["data", "theory"], default="data",
                        help="rescale each trace by its range over the sweep, or by its theoretical range")
    return parser.parse_args().norm


def style_axis(ax, w):
    """Shared x-axis, ticks, fonts and spines."""
    ax.set_xlabel("$w$", fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE, width=1.2, length=5)
    ax.set_xlim(w[0], w[-1])
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.spines[["top", "right"]].set_visible(False)


def draw_traces(ax, w, traces, norm):
    """Plot confidence traces [(label, y, (theory_lo, theory_hi)), ...], each rescaled to [0, 1] (see
    `rescale`) and styled by READOUT_STYLE. Returns the rescaled traces as [(label, y), ...]."""
    scaled = [(label, rescale(y, bounds, norm)) for label, y, bounds in traces]
    for label, y in scaled:
        color, ls = READOUT_STYLE[label]
        ax.plot(w, y, color=color, ls=ls, lw=2.5, label=label)
    style_axis(ax, w)
    ax.set_ylabel("confidence $c[p(x\\,|\\,w)]$", fontsize=LABEL_SIZE)
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.5, 1])
    return scaled


def print_traces(w, traces):
    i0 = np.argmin(np.abs(w))
    for label, y in traces:
        print(f"{label:20s} w=0: {y[i0]:+.4f}  w={w[-1]:g}: {y[-1]:+.4f}")
