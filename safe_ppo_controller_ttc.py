"""
safe_ppo_controller.py
======================
Final Method — PPO + Safety Shield (TTC-based).

This version replaces the raw nearest-distance shield with a
time-to-collision (TTC) based shield, since diagnostics showed
raw distance alone does not cleanly separate crashed vs. safe
episodes (overlapping distributions, n=30 episodes). TTC accounts
for closing speed, not just current distance, and reuses the same
"closing = -vx" convention already used in evaluate.py's compute_ttc().
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
    model_path      : path to models/ppo_final.zip
    normalizer_path : path to models/vecnormalize.pkl
    ttc_threshold   : if predicted time-to-collision with any vehicle is
                       below this (in normalised time units, consistent
                       with evaluate.py's compute_ttc), the shield engages.
    danger_dist     : absolute fallback — if a vehicle is closer than this
                       regardless of closing speed, shield engages too
                       (covers near-zero-closing-speed but very close cases).
    deterministic   : use deterministic actions at eval time
    """

    def __init__(self, model_path, normalizer_path="models/vecnormalize.pkl",
                 ttc_threshold=1.0, danger_dist=0.15, deterministic=True):
        self.model = PPO.load(model_path)
        self.ttc_threshold = ttc_threshold
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

    def _min_ttc(self, observation):
        """Same convention as evaluate.py's compute_ttc: closing = -vx."""
        obs = np.array(observation)
        min_ttc = float("inf")
        min_dist_any_closing_state = float("inf")
        for i in range(1, obs.shape[0]):
            if obs[i, 0] < 0.5:
                continue
            x, y, vx = obs[i, 1], obs[i, 2], obs[i, 3]
            dist = np.sqrt(x**2 + y**2)
            min_dist_any_closing_state = min(min_dist_any_closing_state, dist)
            closing = -vx
            if closing > 0.01:
                ttc = dist / closing
                min_ttc = min(min_ttc, ttc)
        return min_ttc, min_dist_any_closing_state

    def _is_dangerous(self, observation):
        ttc, min_dist = self._min_ttc(observation)
        # Danger if closing fast enough that we'd collide soon, OR
        # if a vehicle is extremely close regardless of closing speed.
        return (ttc < self.ttc_threshold) or (min_dist < self.danger_dist)

    def act(self, observation):
        obs = np.array(observation)[np.newaxis, ...]
        if self.normalizer is not None:
            obs = self.normalizer.normalize_obs(obs)
        proposed, _ = self.model.predict(obs, deterministic=self.deterministic)
        proposed = int(proposed[0])

        shielded = self._is_dangerous(observation)
        action = SLOWER if shielded else proposed

        return action, {
            "controller": "safe_ppo",
            "proposed_action": proposed,
            "shielded": shielded,
        }