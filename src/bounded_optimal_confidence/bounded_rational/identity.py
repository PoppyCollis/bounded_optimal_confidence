"""What a BA state is and which way it leans: the state dict, mirror symmetry of a world, and chirality.

From ITBR_confidence/fixed_points/identity.py and `crossings` from ITBR_confidence/regime_map/branches.py. On a
mirror-symmetric world (configs 1 and 2) a solved cut at +c and at -c are different policies with identical J; `skew`,
the centred mean of p(a), is odd under the mirror and tells them apart.
"""
import numpy as np

from bounded_optimal_confidence.bounded_rational import settings


def crossings(pogw):
    """Indices where argmax_o p(o|w) changes along w.

    Read off the argmax rather than off p(o|w) itself so it means the same thing at a hard
    (beta1 = inf) router and at a soft one: it is the partition the router would induce, which is
    exactly what self_consistency compares against and what the enumeration enumerates.
    """
    assign = np.asarray(pogw, dtype=float).argmax(axis=0)
    return np.flatnonzero(np.diff(assign) != 0)


def state_dict(po, pa, pogw, pago, pagow, pagw):
    """The six components of a BA state, as the dict this package passes around.

    A dict rather than the solver's 5-tuple because `pagw` belongs with the rest here: it is the
    read-out p(a|w) that model fitting consumes, and the whole point of this store is that it stops
    being unrecoverable. Arrays are copied to float64 so a stored state can never alias a solver
    buffer that is about to be overwritten.
    """
    return {"po": np.array(po, dtype=float), "pa": np.array(pa, dtype=float),
            "pogw": np.array(pogw, dtype=float), "pago": np.array(pago, dtype=float),
            "pagow": np.array(pagow, dtype=float), "pagw": np.array(pagw, dtype=float)}


def is_mirror_symmetric(mu_as, tol=1e-9):
    """Is this world exactly mirror-symmetric in w, so the mirror is a symmetry of the problem?

    True when the stimulus means are their own reflection about their midpoint -- configs 1 and 2.
    False on the asymmetric configs (3, 4), where the reflected policy is a genuinely different
    solution with a different J and there is no pair to store.
    """
    mu_as = np.asarray(mu_as, dtype=float)
    centre = 0.5 * (mu_as[0] + mu_as[-1])
    return bool(np.abs((mu_as - centre) + (mu_as - centre)[::-1]).max() < tol)


def skew(pa):
    """The centred mean of p(a). ODD under the mirror: skew(mirror(s)) = -skew(s).

    The primary chirality statistic. It is a linear functional of a converged marginal rather than
    anything read off an argmax, so it is smooth in the betas and does not flicker -- which is why
    it is preferred over the cut position, whose crossings are meaningless on a collapsed router
    (a flat p(o|w) puts its argmax wherever the noise falls; the stored config-1 run has such a
    cell reporting crossings at -73.88 and +28.59 for what is one and the same solution).
    """
    pa = np.asarray(pa, dtype=float)
    return float((np.arange(len(pa)) - (len(pa) - 1) / 2.0) @ pa)


def crossing_centroid(pogw, w_vals):
    """Mean position of the router's crossings, as the MIDPOINT of each crossed pair. Also odd.

    The midpoint, not w_vals[i], and that is not cosmetic. `hard_partition` is right-closed, so the
    naive w <-> -w pairing of cut INDICES is off by one grid point (regime_map measures the
    resulting spurious asymmetry at ~1e-3, against 1.8e-15 under the correct pairing i <-> N-2-i).
    Taking 0.5*(w[i] + w[i+1]) removes that offset exactly: under the array flip a crossing at i
    becomes one at N-2-i, whose midpoint is -(0.5*(w[i] + w[i+1])).

    The fallback statistic, used only where `skew` is inside its tolerance. Returns 0.0 for a
    router with no crossing, which is the right answer -- there is no partition to lean.
    """
    xs = crossings(pogw)
    if not len(xs):
        return 0.0
    w_vals = np.asarray(w_vals, dtype=float)
    return float(np.mean(0.5 * (w_vals[xs] + w_vals[xs + 1])))


def chirality(state, w_vals, symmetric, card=None):
    """(chirality, skew, centroid) for a state. chirality is +1, -1 or 0 (self-mirror).

    Zero on an asymmetric world by construction, since there is no mirror symmetry there to break.

    `skew` decides it; `crossing_centroid` is only consulted when `skew` is inside its tolerance
    AND the state has effective cardinality above 1. That last condition is the guard the flat
    router needs: with p(o|w) = p(o) the argmax is decided by whichever way the noise fell, and the
    stored config-1 run has one such cell reporting crossings at -73.88 and +28.59 for what is one
    and the same solution. A collapsed state is its own mirror; it must not be given a lean by a
    coin flip.

    Both statistics are returned, not just the verdict, so a run's report can census BOTH edges of
    the gap they separate -- the largest residue among self-mirror rows and the smallest genuine
    lean -- and say whether the tolerance is anywhere near being asked a real question. The repo's
    convention is to measure a threshold rather than assert it.
    """
    s = skew(state["pa"])
    c = crossing_centroid(state["pogw"], w_vals)
    if not symmetric:
        return 0, s, c
    if abs(s) >= settings.CHIRALITY_SKEW_TOL:
        return (1 if s > 0 else -1), s, c
    if card is not None and int(card) > 1:
        w_vals = np.asarray(w_vals, dtype=float)
        tol = settings.CHIRALITY_CENTROID_GRID_FRAC * float(abs(w_vals[1] - w_vals[0]))
        if abs(c) >= tol:
            return (1 if c > 0 else -1), s, c
    return 0, s, c
