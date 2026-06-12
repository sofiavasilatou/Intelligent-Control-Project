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
        "intersection-v0",  # MANDATORY: Matches your specific project curriculum requirement
        render_mode=render_mode,
        config={
            "observation": KINEMATIC_OBS,
            "action": {"type": "DiscreteMetaAction"},

            # Longer duration allows the agent room to yield and cross completely
            "duration": 20,

            # Dense reward scheme
            "reward_weights": [1, 0.0, 0.0, 0.1, 0.05, 0.0],
            "normalize_reward": False,       # CRITICAL FIX: Disables early 4-step truncation!
            "collision_reward": -5.0,       # Heavy penalty for crashing
            "high_speed_reward": 1.0,       # Reward for moving fast
            "arrived_reward": 5.0,          # Big bonus for crossing completely

            # Traffic settings (manageable density for stable policy growth)
            "initial_vehicle_count": 5,     
            "spawn_probability": 0.4,

            # Give the ego vehicle clear room to establish a safe path
            "controlled_vehicles": 1,
        },
    )
    return env