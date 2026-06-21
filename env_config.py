"""
env_config.py
=============
Creates the mandatory intersection-v0 environment.

DO NOT modify reward values, termination logic, or random seeds.
Project rules forbid changes to: collision_reward, high_speed_reward,
arrived_reward, normalize_reward, or any environment dynamics.
"""

import gymnasium as gym
import highway_env  # noqa: F401


KINEMATIC_OBS = {
    "type": "Kinematics",
    "vehicles_count": 15,
    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
    "features_range": {
        "x": [-100, 100],
        "y": [-100, 100],
        "vx": [-20, 20],
        "vy": [-20, 20],
    },
    "absolute": False,
    "order": "sorted",
}


def make_env(render_mode=None):
    env = gym.make(
        "intersection-v0",
        render_mode=render_mode,
        config={
            "observation": KINEMATIC_OBS,
            "action": {
                "type": "DiscreteMetaAction",
                "longitudinal": True,
                "lateral": True,
            },

            # ── Rewards: DO NOT CHANGE (project rules) ────────────────────
            "collision_reward":  -5.0,    # original value — do not touch
            "high_speed_reward":  1.0,    # original value — do not touch
            "arrived_reward":     5.0,    # original value — do not touch
            "normalize_reward":   False,  # keep False

            # ── Traffic: realistic density matching hidden test conditions ─
            "initial_vehicle_count": 10,  # default — do not lower
            "spawn_probability":     0.6, # default — do not lower
            "duration":              13,  # slightly longer than default (13)

            "controlled_vehicles": 1,
        },
    )
    return env