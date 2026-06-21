"""
safe_ppo_controller.py
======================
Final Method — PPO + Safety Shield.
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from env_config import make_env

IDLE   = 1
SLOWER = 4


class SafePPOController:
    """
    Parameters
    ----------
    model_path     : path to models/ppo_final.zip
    normalizer_path: path to models/vecnormalize.pkl
    danger_dist    : normalised distance threshold for the shield
    deterministic  : use deterministic actions
    """

    def __init__(self, model_path, normalizer_path="models/vecnormalize.pkl",
                 danger_dist=0.15, deterministic=True):
        self.model = PPO.load(model_path)
        self.danger_dist = danger_dist
        self.deterministic = deterministic
        self.normalizer = None

        if normalizer_path:
            try:
                venv = DummyVecEnv([make_env])
                self.normalizer = VecNormalize.load(normalizer_path, venv)
                self.normalizer.training = False
                self.normalizer.norm_reward = False
                print(f"[SafePPOController] Loaded normalizer from {normalizer_path}")
            except FileNotFoundError:
                print("[SafePPOController] No normalizer found — running without it.")

        print(f"[SafePPOController] Loaded model from {model_path}")

    def reset(self, seed=None):
        pass

    
        # AFTER (entire method):
    def _is_dangerous(self, observation):
        obs    = np.array(observation, dtype=float)
        ego_vx = obs[0, 3]   # ego forward speed (normalised)

        # KEY INSIGHT: if the ego is already committed to crossing (moving fast),
        # do NOT shield — braking mid-intersection is MORE dangerous than continuing.
        # Only apply the shield when the ego is slow (still deciding to enter).
        if ego_vx > 0.6:
            return False, "safe"

        for i in range(1, obs.shape[0]):
            if obs[i, 0] < 0.5:
                continue

            x  = obs[i, 1]
            y  = obs[i, 2]
            vx = obs[i, 3]
            vy = obs[i, 4]

            dist = np.sqrt(x**2 + y**2)

            # Only care about vehicles that are close AND crossing our path
            if dist > 0.20:
                continue

            # Ignore vehicles behind us
            if x < -0.05:
                continue

            # Check if genuinely closing in
            dx = x / (dist + 1e-9)
            dy = y / (dist + 1e-9)
            closing_speed = -((vx - ego_vx) * dx + vy * dy)

            if closing_speed > 0.03:
                return True, "approach"

        return False, "safe"


# AFTER:
    def act(self, observation):
        obs_input = np.array(observation)[np.newaxis, ...]
        if self.normalizer is not None:
            obs_input = self.normalizer.normalize_obs(obs_input)
        proposed, _ = self.model.predict(obs_input, deterministic=self.deterministic)
        proposed = int(proposed[0])

        dangerous, danger_level = self._is_dangerous(observation)

        if dangerous:
            ego_vx = float(np.array(observation)[0, 3])
            action = SLOWER if ego_vx > 0.05 else IDLE
        else:
            action = proposed

        return action, {
            "controller":      "safe_ppo",
            "proposed_action": proposed,
            "shielded":        dangerous,
            "danger_level":    danger_level,
        }