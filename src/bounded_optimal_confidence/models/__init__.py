"""The decision models: the Bayes posterior, attention over structure repertoires, readouts and the simulator.

    bayes       the standard Bayesian decision model's posterior
    attention   repertoires of attentional masks and collapsed inference over them
    readouts    posterior -> continuous confidence
    registry    the paper's ten models (MODELS, get_model)
    simulate    a model's decisions and confidences on a subject's trials
"""
from bounded_optimal_confidence.models.registry import MODELS, Model, get_model
from bounded_optimal_confidence.models.simulate import simulate

__all__ = ["MODELS", "Model", "get_model", "simulate"]
