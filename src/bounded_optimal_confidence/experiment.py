"""Li & Ma (2020) experiment 1: the task constants and the trial data.

Three categories, each a 2-D Gaussian of standard deviation SIG_S around its centre; four stimulus configurations
(category layouts), all collinear on y = 0; four confidence levels; 13 subjects. See data/li_ma_2020/README.md.
"""
from dataclasses import dataclass

import numpy as np
import scipy.io

from bounded_optimal_confidence.paths import DATA_DIR

EXPERIMENT = 1
SUBJECTS = tuple(range(1, 14))
CONFIGS = (1, 2, 3, 4)
N_CATEGORIES = 3
N_CONFIDENCE_LEVELS = 4
SIG_S = 64.0                    # category (stimulus) standard deviation

# category centres per config, as float64 arrays. The y centres are all 0 in experiment 1; they are kept (rather than
# dropping the y dimension) because the Bayes posterior is computed in 2-D exactly as in the original code.
CATEGORY_X = {1: np.array([-96.0, 0.0, 96.0]),
              2: np.array([-128.0, 0.0, 128.0]),
              3: np.array([-96.0, -59.0, 96.0]),
              4: np.array([-96.0, 59.0, 96.0])}
CATEGORY_Y = {cfg: np.zeros(N_CATEGORIES) for cfg in CONFIGS}


@dataclass(frozen=True)
class Trials:
    """One subject's trials in one config. Decisions and confidences are 1-based, as reported."""

    target_x: np.ndarray        # float
    target_y: np.ndarray        # float
    decision: np.ndarray        # int, 1..N_CATEGORIES
    confidence: np.ndarray      # int, 1..N_CONFIDENCE_LEVELS

    def __len__(self):
        return len(self.target_x)


def load_trials(subject, cfg):
    """The trials of `subject` in config `cfg`, in their recorded order."""
    trl = scipy.io.loadmat(DATA_DIR / f"EXP{EXPERIMENT}" / f"S{subject}_log.mat")["trl"]

    def field(name, dtype):
        return trl[name][0, 0].ravel().astype(dtype)

    keep = field("config", np.int64) == cfg
    return Trials(target_x=field("target_x", float)[keep], target_y=field("target_y", float)[keep],
                  decision=field("estD", np.int64)[keep], confidence=field("estC", np.int64)[keep])


def load_subject(subject, cfgs=CONFIGS):
    """[Trials of each config], in `cfgs` order: the blocks a pooled fit joins."""
    return [load_trials(subject, cfg) for cfg in cfgs]
