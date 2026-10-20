"""The paper's ten decision models, defined in one place.

Every model shares the same generative story (`simulate`): a noisy measurement of the target, a category
posterior, Dirichlet decision noise, an argmax decision and a confidence readout. They differ in two places only:

    attention   none: the standard Bayes posterior (`bayes.bayes_posterior`)
                hard: the maximum-evidence structure of a repertoire (beta = inf)
                soft: structure selection at a fitted temperature beta (`attention.posterior`)
    readout     diff, entropy, or top-2 entropy (`readouts`)

Free parameters: b1, b2, b3 (confidence boundaries), lapse, sig_x (sensory noise), alpha (Dirichlet decision noise)
-- k = 6 -- plus log10 beta for the soft-attention models (k = 7). The hard-attention posterior has at most two
non-zero categories, so it is read out with top-2 entropy; the soft one has three, so with full entropy.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    name: str               # identifier used on the command line and in results folder names
    label: str              # the paper's name
    repertoire: str | None  # key of attention.REPERTOIRES; None = no attention (standard Bayes)
    attention: str          # "none", "hard" (beta = inf) or "soft" (beta fitted)
    readout: str            # key of readouts.READOUTS

    @property
    def fits_beta(self):
        return self.attention == "soft"

    @property
    def k(self):
        """Number of free parameters (for the AIC)."""
        return 7 if self.fits_beta else 6

    @property
    def parameters(self):
        return ("b1", "b2", "b3", "lapse", "sig_x", "log10_alpha") + (("log10_beta",) if self.fits_beta else ())


_REPERTOIRE_LABEL = {"sym_top2": "Sym Top-2", "sym_spec": "Sym Spec", "asym_spec_l": "Asym-L", "asym_spec_r": "Asym-R"}


def _attention_model(repertoire, attention):
    return Model(name=f"{repertoire}_{attention}", label=f"{_REPERTOIRE_LABEL[repertoire]} ({attention})",
                 repertoire=repertoire, attention=attention,
                 readout="top2_entropy" if attention == "hard" else "entropy")


MODELS = {m.name: m for m in [
    Model("diff", "Diff", None, "none", "diff"),
    Model("ent", "Ent", None, "none", "entropy"),
    *(_attention_model(rep, "hard") for rep in _REPERTOIRE_LABEL),
    *(_attention_model(rep, "soft") for rep in _REPERTOIRE_LABEL),
]}


def get_model(model):
    """A Model from its name (a Model passes through)."""
    if isinstance(model, Model):
        return model
    try:
        return MODELS[model]
    except KeyError:
        raise ValueError(f"unknown model {model!r}; one of {', '.join(MODELS)}") from None
