"""
safe_ppo_controller.py
======================
Final Method — PPO + Safety Shield.

The shield intercepts the PPO agent's proposed action whenever the raw
observation indicates an imminent collision risk, and substitutes a safe
braking action (SLOWER or IDLE).

Key fix over the original:
  The original _is_dangerous() used `vx <= 0` as the approach criterion.
  At an intersection, the most common threat is a vehicle crossing laterally
  (large vy, near-zero vx), which the old check silently missed.
  The corrected version uses the same 2-D closing-speed projection used in
  the improved rule_based_controller: it detects approach from any direction.
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from env_config import make_env

# ── Action indices ────────────────────────────────────────────────────────────
IDLE   = 1
SLOWER = 4

# ── Feature indices: [presence, x, y, vx, vy, cos_h, sin_h] ─────────────────
F_PRESENCE = 0
F_X        = 1
F_Y        = 2
F_VX       = 3
F_VY       = 4


class SafePPOController:
    """
    Parameters
    ----------
    model_path     : path to models/ppo_final.zip (or ppo_best/best_model.zip)
    normalizer_path: path to models/vecnormalize.pkl
    critical_dist  : normalised distance for emergency stop (any direction)
    danger_dist    : normalised distance for yield zone (closing vehicles only)
    closing_speed_threshold : minimum closing speed to trigger the shield
    deterministic  : use deterministic actions from the PPO policy
    """

    def __init__(
        self,
        model_path,
        normalizer_path="models/vecnormalize.pkl",
        critical_dist=0.10,
        danger_dist=0.20,
        closing_speed_threshold=0.01,
        deterministic=True,
    ):
        self.model                   = PPO.load(model_path)
        self.critical_dist           = critical_dist
        self.danger_dist             = danger_dist
        self.closing_speed_threshold = closing_speed_threshold
        self.deterministic           = deterministic
        self.normalizer              = None

        if normalizer_path:
            try:
                venv = DummyVecEnv([make_env])
                self.normalizer = VecNormalize.load(normalizer_path, venv)
                self.normalizer.training   = False
                self.normalizer.norm_reward = False
                print(f"[SafePPOController] Loaded normalizer from {normalizer_path}")
            except FileNotFoundError:
                print("[SafePPOController] No normalizer found — running without it.")

        print(f"[SafePPOController] Loaded model from {model_path}")

    # ------------------------------------------------------------------
    def reset(self, seed=None):
        pass

    # ------------------------------------------------------------------
    def _is_dangerous(self, observation):
        """
        Returns (is_dangerous, danger_level) where danger_level is one of:
          'critical'  — vehicle inside critical_dist (stop regardless of direction)
          'approach'  — closing vehicle inside danger_dist
          'safe'      — no threat detected

        Uses full 2-D closing-speed projection so lateral threats (crossing
        vehicles) are correctly detected — not just vehicles directly ahead.
        """
        obs    = np.array(observation, dtype=float)
        ego_vx = obs[0, F_VX]

        for i in range(1, obs.shape[0]):
            if obs[i, F_PRESENCE] < 0.5:
                continue

            x  = obs[i, F_X]
            y  = obs[i, F_Y]
            vx = obs[i, F_VX]
            vy = obs[i, F_VY]

            dist = np.sqrt(x**2 + y**2)

            # Critical zone: too close to do anything but stop
            if dist < self.critical_dist:
                return True, "critical"

            # Yield zone: check if the vehicle is genuinely closing in
            if dist < self.danger_dist:
                dx, dy = x / (dist + 1e-9), y / (dist + 1e-9)
                rel_vx = vx - ego_vx
                rel_vy = vy
                closing_speed = -(rel_vx * dx + rel_vy * dy)
                if closing_speed > self.closing_speed_threshold:
                    return True, "approach"

        return False, "safe"

    # ------------------------------------------------------------------
    def act(self, observation):
        # Get PPO's proposed action (using normalised obs if available)
        obs_input = np.array(observation)[np.newaxis, ...]
        if self.normalizer is not None:
            obs_input = self.normalizer.normalize_obs(obs_input)
        proposed, _ = self.model.predict(obs_input, deterministic=self.deterministic)
        proposed = int(proposed[0])

        # Check shield on raw (un-normalised) observation
        dangerous, danger_level = self._is_dangerous(observation)

        if dangerous:
            ego_vx = float(np.array(observation)[0, F_VX])
            if danger_level == "critical":
                action = SLOWER if ego_vx > 0.05 else IDLE
            else:  # approach
                action = SLOWER if ego_vx > 0.15 else IDLE
        else:
            action = proposed

        return action, {
            "controller":      "safe_ppo",
            "proposed_action": proposed,
            "shielded":        dangerous,
            "danger_level":    danger_level,
        }
