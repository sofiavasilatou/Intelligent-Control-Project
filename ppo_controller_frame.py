"""
ppo_controller_frame.py
=======================
PPO Controller with Frame Stacking and Observation Normalization.
"""

import numpy as np
from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from env_config import make_env

N_STACK = 4

class PPOController:
    def __init__(self, model_path, normalizer_path="models/vecnormalize.pkl",
                 deterministic=True):
        if model_path.endswith(".zip"):
            model_path = model_path[:-4]
            
        self.model         = PPO.load(model_path)
        self.deterministic = deterministic
        self.n_stack       = N_STACK
        self.frames        = deque(maxlen=N_STACK)
        self.normalizer    = None

        # Load the normalizer so the AI receives the exact same scale of data it trained on
        if normalizer_path:
            try:
                venv = DummyVecEnv([make_env])
                self.normalizer = VecNormalize.load(normalizer_path, venv)
                self.normalizer.training    = False
                self.normalizer.norm_reward = False
                print(f"[PPOController] Loaded normalizer from {normalizer_path}")
            except FileNotFoundError:
                print("[PPOController] No normalizer found — running without it.")

        print(f"[PPOController] Loaded model from {model_path}")
        print(f"[PPOController] Model obs space: {self.model.observation_space}")

    def reset(self, seed=None):
        self.frames.clear()

    def _get_stacked_obs(self, observation):
        obs = np.array(observation, dtype=np.float32)  # Base shape: (15, 7)

        # 1. Normalize the current frame BEFORE stacking
        if self.normalizer is not None:
            obs_batch = obs[np.newaxis, ...]                     # Temporarily make it (1, 15, 7) for the normalizer
            obs_batch = self.normalizer.normalize_obs(obs_batch) # Apply normalization
            obs = obs_batch[0]                                   # Strip the batch dimension back to (15, 7)

        # 2. Frame Stacking Logic
        if len(self.frames) == 0:
            # If this is the first step, copy the initial frame N times
            for _ in range(self.n_stack):
                self.frames.append(obs.copy())
        else:
            # Otherwise, just push the newest frame (the oldest drops off automatically)
            self.frames.append(obs.copy())

        # 3. Concatenate horizontally to make the wide (15, 28) array the network expects
        stacked = np.concatenate(list(self.frames), axis=1)   # (15, 28)
        
        # 4. Add the batch dimension back for the model prediction
        return stacked[np.newaxis, ...]                       # (1, 15, 28)

    def act(self, observation):
        stacked_obs = self._get_stacked_obs(observation)
        action, _   = self.model.predict(stacked_obs, deterministic=self.deterministic)
        
        return int(action[0]), {"controller": "ppo", "shielded": False}