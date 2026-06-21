"""
ppo_controller.py
=================
Meaningful Competitor — Trained PPO agent with frame stacking.
"""

import numpy as np
from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from env_config import make_env

N_STACK = 4   # must match what was used in train_ppo.py


class PPOController:
    """
    Parameters
    ----------
    model_path      : path to models/ppo_final.zip (or without .zip)
    normalizer_path : path to models/vecnormalize.pkl
    deterministic   : use deterministic actions at eval time
    """

    def __init__(self, model_path, normalizer_path="models/vecnormalize.pkl",
                 deterministic=True):
        # Strip .zip if present — SB3 adds it automatically
        if model_path.endswith(".zip"):
            model_path = model_path[:-4]
        self.model         = PPO.load(model_path)
        self.deterministic = deterministic
        self.normalizer    = None
        self.n_stack       = N_STACK
        self.frames        = deque(maxlen=N_STACK)

        if normalizer_path:
            try:
                venv = DummyVecEnv([make_env])
                self.normalizer = VecNormalize.load(normalizer_path, venv)
                self.normalizer.training   = False
                self.normalizer.norm_reward = False
                print(f"[PPOController] Loaded normalizer from {normalizer_path}")
            except FileNotFoundError:
                print("[PPOController] No normalizer found — running without it.")

        print(f"[PPOController] Loaded model from {model_path}")

    def reset(self, seed=None):
        """Clear the frame buffer at the start of each episode."""
        self.frames.clear()

    def _get_stacked_obs(self, observation):
        """
        Normalize the current observation and stack it with the last
        N_STACK-1 frames. On the first step of an episode the buffer
        is filled with copies of the first observation (cold start).
        """
        obs = np.array(observation, dtype=np.float32).flatten()

        # Normalize using the loaded VecNormalize statistics
        if self.normalizer is not None:
            obs_batch = obs[np.newaxis, ...]
            obs_batch = self.normalizer.normalize_obs(obs_batch)
            obs = obs_batch[0]

        # Fill buffer on first step
        if len(self.frames) == 0:
            for _ in range(self.n_stack):
                self.frames.append(obs)
        else:
            self.frames.append(obs)

        # Concatenate frames → shape (n_stack * obs_size,)
        stacked = np.concatenate(list(self.frames), axis=0)
        return stacked[np.newaxis, ...]   # add batch dim

    def act(self, observation):
        stacked_obs = self._get_stacked_obs(observation)
        action, _   = self.model.predict(stacked_obs, deterministic=self.deterministic)
        return int(action[0]), {"controller": "ppo_frame", "shielded": False}
