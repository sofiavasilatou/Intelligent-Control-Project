"""
ppo_controller.py
=================
Controller wrapper for a trained PPO model.
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    VecNormalize,
    DummyVecEnv
)

from env_config_myrs import make_env


class PPOController:

    def __init__(
        self,
        model_path,
        normalizer_path="models/vecnormalize.pkl",
        deterministic=True
    ):

        self.model = PPO.load(model_path)

        self.deterministic = deterministic
        self.normalizer = None

        try:

            venv = DummyVecEnv([make_env])

            self.normalizer = VecNormalize.load(
                normalizer_path,
                venv
            )

            self.normalizer.training = False
            self.normalizer.norm_reward = False

            print(
                f"[PPO] Loaded normalizer: {normalizer_path}"
            )

        except FileNotFoundError:

            print(
                "[PPO] No normalizer found"
            )

        print(
            f"[PPO] Loaded model: {model_path}"
        )

    def reset(self, seed=None):
        pass


    def act(self, observation):

        obs = np.array(
            observation
        )[np.newaxis, ...]

        if self.normalizer is not None:
            obs = self.normalizer.normalize_obs(
                obs
            )

        action, _ = self.model.predict(
            obs,
            deterministic=self.deterministic
        )

        return int(action[0]), {
            "controller": "ppo",
            "shielded": False
        }