"""
env_config.py
=============
Creates the mandatory intersection-v0 environment with custom dense rewards.
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
            "action": {"type": "DiscreteMetaAction"},

            "duration": 20,

            # DELETE this line — it does nothing, _agent_rewards() never reads it
            # "reward_weights": [1, 0.0, 0.0, 0.1, 0.05, 0.0],

            "normalize_reward": False,
            "collision_reward": -10.0,      # was -5.0 — make crashing dominate speed bonus
            "high_speed_reward": 0.4,       # was 1.0 — reduce per-step speed incentive
            "arrived_reward": 5.0,          # fine as-is

            "initial_vehicle_count": 5,
            "spawn_probability": 0.4,
            "controlled_vehicles": 1,
        },
    )
    return env