"""Scoring a simulated model against a subject's reports.

    scoring    p(decision, confidence level) from a simulation; the clean NLL on a large fresh draw
    profile    the confidence boundaries b1-b3 and the lapse minimised out of one simulation
"""
from bounded_optimal_confidence.likelihood.profile import ProfileResult, profile_b_and_lapse
from bounded_optimal_confidence.likelihood.scoring import clean_nll, report_joint_probs

__all__ = ["ProfileResult", "clean_nll", "profile_b_and_lapse", "report_joint_probs"]
