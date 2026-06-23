import numpy as np

SLOWER = 0
IDLE   = 1
FASTER = 2

F_PRESENCE = 0
F_X        = 1
F_Y        = 2
F_VX       = 3
F_VY       = 4

class RuleBasedController:
    def reset(self, seed=None):
        pass

    def act(self, observation):
        obs = np.array(observation, dtype=float)
        ego_vx = obs[0, F_VX]

        danger = False
        n_threats = 0

        for i in range(1, obs.shape[0]):
            if obs[i, F_PRESENCE] < 0.5:
                continue
                
            x  = obs[i, F_X]
            y  = obs[i, F_Y]
            vx = obs[i, F_VX]
            vy = obs[i, F_VY]

            # Ignore cars behind us
            if x < -0.05:
                continue

            # 1. REAR-END PREVENTION (Same-lane traffic)
            # If a car is right in front of us (y is near 0) and we are closing the gap (vx is negative)
            if abs(y) < 0.05 and x < 0.15:
                if vx < -0.01: 
                    danger = True
                    n_threats += 1
                    break

            # 2. T-BONE PREVENTION (Cross traffic)
            # If a car is moving laterally towards our lane (y and vy have opposite signs)
            if abs(y) < 0.35 and (y * vy < 0):
                
                # Time to cross our lane (Distance scale 100, Velocity scale 20 -> Ratio 5.0)
                t_cross_seconds = abs(y / (vy - 1e-9)) * 5.0 
                
                # If they will cross our lane in less than 2.5 seconds...
                if t_cross_seconds < 2.5:
                    
                    # AND they are within 15 meters of our front bumper!
                    if x < 0.15:
                        danger = True
                        n_threats += 1
                        break

        # Action Execution
        if danger:
            # Brake if moving, idle if stopped
            action = SLOWER if ego_vx > 0.01 else IDLE
        else:
            # Coast or Accelerate up to a safe crossing speed (0.35 = ~7 m/s)
            action = FASTER if ego_vx < 0.35 else IDLE

        return action, {
            "controller": "rule_based",
            "danger":     danger,
            "n_threats":  n_threats,
        }