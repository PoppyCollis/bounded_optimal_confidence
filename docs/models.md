# The decision models

Code: `src/bounded_optimal_confidence/models/`. Notation in the code: category *c*, measurement *x*, structure φ.
The paper writes the latent state as *x* and the observation as *w*.

## The task

Each trial shows a target dot drawn from one of three categories. Each category is a 2-D Gaussian with standard deviation σ_s = 64
around its centre (`experiment.CATEGORY_X`; all centres sit at y = 0 in experiment 1). The subject reports a category
and a confidence level from 1 to 4. A model has to predict the joint distribution of (decision, confidence level) on each trial.

## The generative model shared by all ten

On each trial and each simulated replicate (`models/simulate.py`):

1. **Measurement.** (x, y) = target + N(0, σ_x²) in each dimension. σ_x is the sensory noise, a free parameter.
2. **Bayes posterior** (`models/bayes.py`). The observer knows the layout, so a category-c target is measured as
   N((ux_c, uy_c), σ_p²) with σ_p² = σ_s² + σ_x². With a flat prior,
   p(c | x, y) ∝ exp(−[(x − ux_c)² + (y − uy_c)²] / 2σ_p²).
3. **Attention** (`models/attention.py`). Changes the posterior; the next section has the details.
4. **Decision noise.** p̃ ~ Dirichlet(α p), and the decision is d = argmax p̃. α is a free parameter, and α = ∞ means no noise.
   A category with probability 0 stays at 0.
5. **Confidence** (`models/readouts.py`). A readout of p̃ gives a number in [0, 1], and the boundaries b1 ≤ b2 ≤ b3 bin it
   into the four levels.

The likelihood also mixes in a uniform lapse over all (decision, level) cells; see `docs/fitting.md`.

## Attention: collapsed inference over a repertoire of masks

A repertoire is a set of binary masks φ over the three categories. Each mask is a sparse prior: p(c | φ) is uniform on the
categories the mask keeps. Which mask applies is inferred from the measurement:

    p(x | φ)      = Σ_c p(c | φ) p(x | c)            evidence for structure φ
    p_β(φ | x)    ∝ p(φ) p(x | φ)^β                   structure selection
    p(c | x, φ)   = p(c | φ) p(x | c) / p(x | φ)
    p(c | x)      = Σ_φ p_β(φ | x) p(c | x, φ)         the attended posterior

p(φ) is flat over categories. β tempers the evidence:

- **hard attention**, β = ∞: only the maximum-evidence structure is used. The switch points between structures follow
  from the evidence and move with the layout and with σ_p. The function `attention.cuts` computes them.
- **soft attention**, β fitted: a mixture of the structures. On a repertoire that partitions the categories, β = 1 gives back the
  Bayes posterior exactly.

The four repertoires (`attention.REPERTOIRES`; columns are categories left to right):

| repertoire | structures | in words |
|---|---|---|
| `sym_top2` | {middle, right}, {left, middle} | drop one outer category |
| `sym_spec` | {left, right}, {middle} | the outer pair, or the middle alone |
| `asym_spec_l` | {left}, {middle, right} | the left category alone, or the other two |
| `asym_spec_r` | {right}, {left, middle} | the right category alone, or the other two |

Hard Sym Top-2 is top-2 truncation of the Bayes posterior (`attention.top2_truncate`): keep the two most probable
categories and renormalise. Away from the tie at x = 0 this equals hard attention with `sym_top2`, and
`tests/test_attention.py` checks it. The paper's fit used the truncation form, so this package does too.

## Readouts

| readout | confidence | used by |
|---|---|---|
| `diff` | p(1st) − p(2nd) | Diff |
| `entropy` | 1 − H(p) / log₂ 3 | Ent, the soft-attention models |
| `top2_entropy` | 1 − H(top-2 renormalised), in bits | the hard-attention models |

Hard attention leaves at most two non-zero categories, so full entropy on those posteriors is just the two-category entropy
rescaled onto [1 − 1/log₂ 3, 1]. The hard models read it out on the two-category scale.

## The ten models (`models/registry.py`)

| model | label | attention | readout | k |
|---|---|---|---|---|
| `diff` | Diff | none | diff | 6 |
| `ent` | Ent | none | entropy | 6 |
| `sym_top2_hard`, `sym_spec_hard`, `asym_spec_l_hard`, `asym_spec_r_hard` | Sym Top-2, Sym Spec, Asym-L, Asym-R (hard) | β = ∞ | top-2 entropy | 6 |
| `sym_top2_soft`, `sym_spec_soft`, `asym_spec_l_soft`, `asym_spec_r_soft` | … (soft) | β fitted | entropy | 7 |

The free parameters are b1, b2, b3, lapse, σ_x and α, plus log₁₀ β for the soft models. The nesting matters for
`scripts/summarise_fits.py`. A soft model contains its hard twin at β = ∞. A soft model on a partition repertoire also
contains Ent at β = 1, since that is the Bayes posterior read out with entropy.
