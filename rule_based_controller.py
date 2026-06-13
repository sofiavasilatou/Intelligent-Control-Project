"""
rule_based_controller.py
========================
Baseline 2 — Rule-Based / Scripted Controller for intersection-v0.

Logic (priority order):
  1. CRITICAL DANGER  — any present vehicle within `critical_dist` regardless
                        of heading → brake hard (SLOWER) or stop (IDLE).
  2. APPROACH DANGER  — any present vehicle within `danger_dist` that is
                        closing in (relative speed positive toward ego) →
                        slow down.
  3. CLEAR            — no nearby threat → accelerate (FASTER).

Key fixes over the original controller
---------------------------------------
- Danger detection no longer requires `vx <= 0`. At an intersection, crossing
  vehicles approach from the side with large vy but near-zero vx, so the old
  check silently missed the most common collision scenario.
- Two distance thresholds:  critical_dist (emergency stop zone) and
  danger_dist  (yield zone). This gives smoother, more human-like behaviour.
- Closing-speed check uses the full 2-D relative velocity projected onto the
  ego-to-vehicle vector, rather than just vx. This correctly identifies
  vehicles approaching from any direction.
- Action space constants match the intersection-v0 DiscreteMetaAction set
  with both longitudinal=True and lateral=True enabled:
      0 → LANE_LEFT
      1 → IDLE
      2 → LANE_RIGHT
      3 → FASTER
      4 → SLOWER
"""

import numpy as np

# ── Action indices ────────────────────────────────────────────────────────────
LANE_LEFT  = 0
IDLE       = 1
LANE_RIGHT = 2
FASTER     = 3
SLOWER     = 4

# ── Feature indices in the kinematic observation ──────────────────────────────
# [presence, x, y, vx, vy, cos_h, sin_h]
F_PRESENCE = 0
F_X        = 1
F_Y        = 2
F_VX       = 3
F_VY       = 4


class RuleBasedController:
    """
    Rule-based controller tuned for the intersection-v0 benchmark.

    Parameters
    ----------
    critical_dist : float
        Normalised ego-centric distance inside which the ego must stop
        immediately, regardless of closing direction.  Corresponds to roughly
        10-15 m in physical units given the [-100, 100] feature range.
    danger_dist : float
        Normalised distance inside which a closing vehicle triggers a yield
        (slow-down) response.  Roughly 20-25 m in physical units.
    closing_speed_threshold : float
        Minimum normalised closing speed (dot product of relative velocity with
        unit vector from ego to other vehicle) to classify a vehicle as
        "approaching".  Filters out vehicles that are present but moving away.
    """

    def __init__(
        self,
        critical_dist: float = 0.10,
        danger_dist:   float = 0.20,
        closing_speed_threshold: float = 0.01,
    ):
        self.critical_dist            = critical_dist
        self.danger_dist              = danger_dist
        self.closing_speed_threshold  = closing_speed_threshold

    # ------------------------------------------------------------------
    def reset(self, seed=None):
        pass  # stateless controller — nothing to reset

    # ------------------------------------------------------------------
    def act(self, observation):
        """
        Parameters
        ----------
        observation : array-like, shape (N, 7)
            Row 0   → ego vehicle (features are in ego-centric frame).
            Row 1…N → other vehicles (ego-relative position/velocity).
            Features per row: [presence, x, y, vx, vy, cos_h, sin_h]

        Returns
        -------
        action : int
            One of LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER.
        info   : dict
            Diagnostic fields (used for logging only, not fed back to env).
        """
        obs = np.array(observation, dtype=float)

        # Ego speed along its own heading (row 0, vx in ego frame ≈ forward speed)
        ego_vx = obs[0, F_VX]

        critical_danger = False
        approach_danger = False
        min_dist        = float("inf")
        n_threats       = 0

        for i in range(1, obs.shape[0]):
            if obs[i, F_PRESENCE] < 0.5:   # empty slot — skip
                continue

            x   = obs[i, F_X]
            y   = obs[i, F_Y]
            vx  = obs[i, F_VX]
            vy  = obs[i, F_VY]

            dist = np.sqrt(x**2 + y**2)
            min_dist = min(min_dist, dist)

            # ── Critical zone: stop regardless of approach direction ──────
            if dist < self.critical_dist:
                critical_danger = True
                n_threats += 1
                continue   # no need to check closing speed

            # ── Yield zone: check whether the vehicle is closing in ───────
            if dist < self.danger_dist:
                # Unit vector FROM ego TOWARD the other vehicle
                dx, dy        = x / (dist + 1e-9), y / (dist + 1e-9)
                # Relative velocity of the other vehicle in ego frame
                # (ego velocity in ego frame is approximately [ego_vx, 0])
                rel_vx = vx - ego_vx
                rel_vy = vy           # ego lateral velocity ≈ 0 in ego frame
                # Closing speed = projection of relative velocity onto approach dir
                closing_speed = -(rel_vx * dx + rel_vy * dy)
                if closing_speed > self.closing_speed_threshold:
                    approach_danger = True
                    n_threats += 1

        # ── Decision logic ────────────────────────────────────────────────
        if critical_danger:
            # Emergency: stop or hold if already stopped
            action = SLOWER if ego_vx > 0.05 else IDLE
        elif approach_danger:
            # Yield: slow down
            action = SLOWER if ego_vx > 0.15 else IDLE
        else:
            # No threat detected: accelerate through the intersection
            action = FASTER

        return action, {
            "controller":      "rule_based",
            "critical_danger": critical_danger,
            "approach_danger": approach_danger,
            "n_threats":       n_threats,
            "min_dist":        min_dist if min_dist < float("inf") else None,
            "shielded":        False,
        }