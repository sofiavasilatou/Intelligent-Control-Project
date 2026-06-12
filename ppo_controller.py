"""
ppo_controller.py
=================
Meaningful Competitor — Trained PPO agent.
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from env_config import make_env


class PPOController:
    """
    Parameters
    ----------
    model_path     : path to models/ppo_final.zip
    normalizer_path: path to models/vecnormalize.pkl  (pass None to skip)
    deterministic  : use deterministic actions at eval time
    """

    def __init__(self, model_path, normalizer_path="models/vecnormalize.pkl", deterministic=True):
        self.model = PPO.load(model_path)
        self.deterministic = deterministic
        self.normalizer = None

        if normalizer_path:
            try:
                venv = DummyVecEnv([make_env])
                self.normalizer = VecNormalize.load(normalizer_path, venv)
                self.normalizer.training = False
                self.normalizer.norm_reward = False
                print(f"[PPOController] Loaded normalizer from {normalizer_path}")
            except FileNotFoundError:
                print("[PPOController] No normalizer found — running without it.")

        print(f"[PPOController] Loaded model from {model_path}")

    def reset(self, seed=None):
        pass

    def act(self, observation):
        obs = np.array(observation)[np.newaxis, ...]   # add batch dim
        if self.normalizer is not None:
            obs = self.normalizer.normalize_obs(obs)
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return int(action[0]), {"controller": "ppo"}