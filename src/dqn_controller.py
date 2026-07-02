import numpy as np
from stable_baselines3 import DQN


class DQNController:
    

    def __init__(self, model_path, deterministic=True):
        # Strip .zip if present — SB3 adds it automatically
        if model_path.endswith(".zip"):
            model_path = model_path[:-4]
        self.model         = DQN.load(model_path)
        self.deterministic = deterministic
        print(f"[DQNController] Loaded model from {model_path}")

    def reset(self, seed=None):
        pass  

    def act(self, observation):
       
        obs = np.array(observation, dtype=np.float32)[np.newaxis, ...]
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return int(action[0]), {"controller": "dqn", "shielded": False}