"""
random_controller.py
====================
Baseline 1 — Random Policy.
Picks a random action every step. Lower-bound reference only.
"""

import numpy as np


class RandomController:
    def __init__(self, n_actions=5):
        self.n_actions = n_actions
        self.rng = np.random.default_rng()

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def act(self, observation):
        action = int(self.rng.integers(0, self.n_actions))
        return action, {"controller": "random"}
