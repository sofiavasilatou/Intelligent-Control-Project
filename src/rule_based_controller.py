import numpy as np

SLOWER = 0
IDLE   = 1
FASTER = 2

# Feature indices: [presence, x, y, vx, vy, cos_h, sin_h]
F_PRESENCE = 0
F_X        = 1
F_Y        = 2
F_VX       = 3
F_VY       = 4


class RuleBasedController:
   
    def __init__(
        self,
        critical_dist: float = 0.10,
        danger_dist:   float = 0.20,
        closing_speed_threshold: float = 0.01,
    ):
        self.critical_dist           = critical_dist
        self.danger_dist             = danger_dist
        self.closing_speed_threshold = closing_speed_threshold

    def reset(self, seed=None):
        pass  # stateless

    def act(self, observation):
        obs    = np.array(observation, dtype=float)
        ego_vx = obs[0, F_VX]

        critical_danger = False
        approach_danger = False
        n_threats       = 0

        for i in range(1, obs.shape[0]):
            if obs[i, F_PRESENCE] < 0.5:
                continue

            x  = obs[i, F_X]
            y  = obs[i, F_Y]
            vx = obs[i, F_VX]
            vy = obs[i, F_VY]

            dist = np.sqrt(x**2 + y**2)

            # Critical zone: stop regardless of direction
            if dist < self.critical_dist:
                critical_danger = True
                n_threats += 1
                continue

            # Yield zone: check if vehicle is closing in
            if dist < self.danger_dist:
                dx = x / (dist + 1e-9)
                dy = y / (dist + 1e-9)
                rel_vx = vx - ego_vx
                rel_vy = vy
                closing_speed = -(rel_vx * dx + rel_vy * dy)
                if closing_speed > self.closing_speed_threshold:
                    approach_danger = True
                    n_threats += 1

        # Decision
        if critical_danger:
            action = SLOWER if ego_vx > 0.05 else IDLE
        elif approach_danger:
            action = SLOWER if ego_vx > 0.15 else IDLE
        else:
            action = FASTER

        return action, {
            "controller":      "rule_based",
            "critical_danger": critical_danger,
            "approach_danger": approach_danger,
            "n_threats":       n_threats,
            "shielded":        False,
        }