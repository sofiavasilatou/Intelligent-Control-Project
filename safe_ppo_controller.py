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
                 danger_dist=0.35, deterministic=True):
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

    def _is_dangerous(self, observation):
        obs = np.array(observation)
        for i in range(1, obs.shape[0]):
            if obs[i, 0] < 0.5:
                continue
            x, y, vx = obs[i, 1], obs[i, 2], obs[i, 3]
            dist = np.sqrt(x**2 + y**2)
            if dist < self.danger_dist and vx <= 0.0:
                return True
        return False

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