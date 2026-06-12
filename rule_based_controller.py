"""
rule_based_controller.py
========================
Baseline 2 — Rule-Based / Scripted Controller.

Logic:
  - If any nearby vehicle is close AND approaching → brake / idle.
  - Otherwise → accelerate.

DiscreteMetaAction indices (highway-env default):
    0 → LANE_LEFT
    1 → IDLE
    2 → LANE_RIGHT
    3 → FASTER
    4 → SLOWER
"""

import numpy as np

IDLE   = 1
FASTER = 3
SLOWER = 4


class RuleBasedController:
    """
    Parameters
    ----------
    danger_dist : float
        Normalised distance below which another vehicle is considered dangerous.
    """

    def __init__(self, danger_dist=0.25):
        self.danger_dist = danger_dist

    def reset(self, seed=None):
        pass  # stateless

    def act(self, observation):
        """
        observation shape: (N, F)
          Row 0     → ego vehicle
          Row 1..N  → other vehicles
        Features: [presence, x, y, vx, vy, cos_h, sin_h]
        """
        obs = np.array(observation)
        ego_vx = obs[0, 3]

        danger = False
        for i in range(1, obs.shape[0]):
            if obs[i, 0] < 0.5:      # slot empty
                continue
            x  = obs[i, 1]
            y  = obs[i, 2]
            vx = obs[i, 3]
            dist = np.sqrt(x**2 + y**2)
            if dist < self.danger_dist and vx <= 0.0:
                danger = True
                break

        if danger:
            action = SLOWER if ego_vx > 0.1 else IDLE
        else:
            action = FASTER

        return action, {"controller": "rule_based", "danger": danger}
