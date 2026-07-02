import numpy as np
from stable_baselines3 import DQN

IDLE   = 1
SLOWER = 4

F_PRESENCE = 0
F_X        = 1
F_Y        = 2
F_VX       = 3
F_VY       = 4


class SafeDQNController:
    """
    Parameters
    ----------
    model_path    : path to models/dqn_final.zip or dqn_best/best_model
    danger_dist   : normalised distance threshold for the shield
    deterministic : use greedy actions from DQN
    """

    def __init__(self, model_path, danger_dist=0.15, deterministic=True):
        if model_path.endswith(".zip"):
            model_path = model_path[:-4]
        self.model         = DQN.load(model_path)
        self.danger_dist   = danger_dist
        self.deterministic = deterministic
        print(f"[SafeDQNController] Loaded model from {model_path}")

    def reset(self, seed=None):
        pass  # stateless

    def _is_dangerous(self, observation):
        obs    = np.array(observation, dtype=float)
        ego_vx = obs[0, F_VX]

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

            # 1. Critical zone: anything very close → stop immediately
            if dist < 0.07:
                return True, "critical"

            # 2. Tailgating: vehicle directly ahead, ego is faster
            if x > 0 and abs(y) < 0.08 and dist < 0.12:
                if ego_vx > vx + 0.05:
                    return True, "tailgating"

            # 3. Lateral approach: closing vehicle within danger zone
            if dist < self.danger_dist:
                dx = x / (dist + 1e-9)
                dy = y / (dist + 1e-9)
                closing_speed = -((vx - ego_vx) * dx + vy * dy)
                if closing_speed > 0.03:
                    return True, "approach"

        return False, "safe"

    def act(self, observation):
        obs         = np.array(observation, dtype=np.float32)[np.newaxis, ...]
        proposed, _ = self.model.predict(obs, deterministic=self.deterministic)
        proposed    = int(proposed[0])

        dangerous, danger_level = self._is_dangerous(observation)

        if dangerous:
            ego_vx = float(np.array(observation)[0, F_VX])
            action = SLOWER if ego_vx > 0.05 else IDLE
        else:
            action = proposed

        return action, {
            "controller":      "safe_dqn",
            "proposed_action": proposed,
            "shielded":        dangerous,
            "danger_level":    danger_level,
        }