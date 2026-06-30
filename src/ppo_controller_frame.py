"""PPO Controller with Frame Stacking and Observation Normalization"""

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
        obs = np.array(observation, dtype=np.float32)  

        # 1. Normalize the current frame BEFORE stacking
        if self.normalizer is not None:
            obs_batch = obs[np.newaxis, ...]                     
            obs_batch = self.normalizer.normalize_obs(obs_batch) 
            obs = obs_batch[0]                                   

        if len(self.frames) == 0:
            for _ in range(self.n_stack):
                self.frames.append(obs.copy())
        else:
            self.frames.append(obs.copy())

        stacked = np.concatenate(list(self.frames), axis=1)   
        
        
        return stacked[np.newaxis, ...]                      

    def act(self, observation):
        stacked_obs = self._get_stacked_obs(observation)
        action, _   = self.model.predict(stacked_obs, deterministic=self.deterministic)
        
        return int(action[0]), {"controller": "ppo", "shielded": False}