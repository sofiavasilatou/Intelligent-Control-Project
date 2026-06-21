"""
safe_ppo_controller.py
======================
Final Method — PPO + Safety Shield with frame stacking.
"""

import numpy as np
from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from env_config import make_env

# Action indices
IDLE   = 1
SLOWER = 4

# Feature indices: [presence, x, y, vx, vy, cos_h, sin_h]
F_PRESENCE = 0
F_X        = 1
F_Y        = 2
F_VX       = 3
F_VY       = 4

N_STACK = 4   # must match train_ppo.py


class SafePPOController:
    """
    Parameters
    ----------
    model_path      : path to models/ppo_final.zip
    normalizer_path : path to models/vecnormalize.pkl
    danger_dist     : normalised distance threshold for the shield
    deterministic   : use deterministic actions
    """

    def __init__(self, model_path, normalizer_path="models/vecnormalize.pkl",
                 danger_dist=0.15, deterministic=True):
        # Strip .zip if present
        if model_path.endswith(".zip"):
            model_path = model_path[:-4]
        self.model         = PPO.load(model_path)
        self.danger_dist   = danger_dist
        self.deterministic = deterministic
        self.normalizer    = None
        self.n_stack       = N_STACK
        self.frames        = deque(maxlen=N_STACK)

        if normalizer_path:
            try:
                venv = DummyVecEnv([make_env])
                self.normalizer = VecNormalize.load(normalizer_path, venv)
                self.normalizer.training    = False
                self.normalizer.norm_reward = False
                print(f"[SafePPOController] Loaded normalizer from {normalizer_path}")
            except FileNotFoundError:
                print("[SafePPOController] No normalizer found — running without it.")

        print(f"[SafePPOController] Loaded model from {model_path}")

    def reset(self, seed=None):
        """Clear the frame buffer at the start of each episode."""
        self.frames.clear()

    def _get_stacked_obs(self, observation):
        """Normalize + stack frames, same as PPOController."""
        obs = np.array(observation, dtype=np.float32).flatten()

        if self.normalizer is not None:
            obs_batch = obs[np.newaxis, ...]
            obs_batch = self.normalizer.normalize_obs(obs_batch)
            obs = obs_batch[0]

        if len(self.frames) == 0:
            for _ in range(self.n_stack):
                self.frames.append(obs)
        else:
            self.frames.append(obs)

        stacked = np.concatenate(list(self.frames), axis=0)
        return stacked[np.newaxis, ...]

    def _is_dangerous(self, observation):
        """
        Check raw (un-normalised) observation for imminent danger.
        Only shields when ego is slow (not yet committed to crossing).
        Returns (is_dangerous, danger_level).
        """
        obs    = np.array(observation, dtype=float)
        ego_vx = obs[0, F_VX]

        # If already committed to crossing, do not override
        if ego_vx > 0.6:
            return False, "safe"

        for i in range(1, obs.shape[0]):
            if obs[i, F_PRESENCE] < 0.5:
                continue

            x  = obs[i, F_X]
            y  = obs[i, F_Y]
            vx = obs[i, F_VX]
            vy = obs[i, F_VY]

            dist = np.sqrt(x**2 + y**2)

            # Critical zone: anything very close
            if dist < 0.07:
                return True, "critical"

            # Tailgating: vehicle directly ahead, ego is faster
            if x > 0 and abs(y) < 0.08 and dist < 0.12:
                if ego_vx > vx + 0.05:
                    return True, "tailgating"

            # Lateral approach: closing vehicle from any direction
            if dist < self.danger_dist:
                dx = x / (dist + 1e-9)
                dy = y / (dist + 1e-9)
                closing_speed = -((vx - ego_vx) * dx + vy * dy)
                if closing_speed > 0.03:
                    return True, "approach"

        return False, "safe"

    def act(self, observation):
        # Get PPO action using stacked frames
        stacked_obs      = self._get_stacked_obs(observation)
        proposed, _      = self.model.predict(stacked_obs, deterministic=self.deterministic)
        proposed         = int(proposed[0])

        # Check shield on raw observation
        dangerous, danger_level = self._is_dangerous(observation)

        if dangerous:
            ego_vx = float(np.array(observation)[0, F_VX])
            action = SLOWER if ego_vx > 0.05 else IDLE
        else:
            action = proposed

        return action, {
            "controller":      "safe_ppo_frame",
            "proposed_action": proposed,
            "shielded":        dangerous,
            "danger_level":    danger_level,
        }
