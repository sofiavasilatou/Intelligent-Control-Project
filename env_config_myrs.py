"""
Project 4: Safe Autonomous Driving at Intersections
Shared environment configuration for intersection-v0.

Import this in any script (random baseline, PPO training/eval, etc.) so
every controller is evaluated under the exact same observation and action
settings, as required by the project rules.
"""

# ---------------------------------------------------------------------
# Observation variant 1: Low-dimensional kinematic observation (default)
# ---------------------------------------------------------------------
KINEMATIC_OBS = {
    "type": "Kinematics",

    # Number of vehicles included in the observation, including ego.
    "vehicles_count": 15,

    # Per-vehicle features. "presence" indicates whether a vehicle slot
    # is occupied.
    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],

    # Normalize features to approximately bounded ranges.
    "features_range": {
        "x": [-100, 100],
        "y": [-100, 100],
        "vx": [-20, 20],
        "vy": [-20, 20],
    },

    # Use ego-centric coordinates.
    "absolute": False,

    # Keep a fixed-size observation even when fewer vehicles are nearby.
    "order": "sorted",
}


# ---------------------------------------------------------------------
# Observation variant 2: Occupancy-grid observation (alternative)
# ---------------------------------------------------------------------
OCCUPANCY_GRID_OBS = {
    "type": "OccupancyGrid",

    # Number of nearby vehicles considered when filling the grid.
    "vehicles_count": 15,

    # Per-cell features.
    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],

    "features_range": {
        "x": [-100, 100],
        "y": [-100, 100],
        "vx": [-20, 20],
        "vy": [-20, 20],
    },

    # Grid around the ego vehicle, in meters.
    "grid_size": [[-30, 30], [-30, 30]],

    # Cell size, in meters.
    "grid_step": [5, 5],

    # Ego-centric coordinates.
    "absolute": False,
}


# ---------------------------------------------------------------------
# Action space: discrete high-level meta-actions
#
# IMPORTANT: intersection-v0's actual default action config (see
# IntersectionEnv.default_config() in highway_env/envs/intersection_env.py)
# is NOT the full 5-action DiscreteMetaAction set. It restricts the agent
# to longitudinal (speed) control only and disables lateral/steering
# meta-actions, with 3 discrete target speeds:
#
#   "action": {
#       "type": "DiscreteMetaAction",
#       "longitudinal": True,
#       "lateral": False,
#       "target_speeds": [0, 4.5, 9],
#   }
#
# In that mode the action space effectively has 3 actions
# (SLOWER / IDLE / FASTER over 3 target speeds), not the 5-action
# (LANE_LEFT / IDLE / LANE_RIGHT / FASTER / SLOWER) set you'd get from
# DiscreteMetaAction's own defaults elsewhere in highway-env (e.g. on
# highway-v0). The project PDF's minimal example just passes
# {"type": "DiscreteMetaAction"} with no longitudinal/lateral/target_speeds
# keys, which falls back on DiscreteMetaAction's internal defaults
# (5 actions, lateral lane-change enabled) rather than IntersectionEnv's
# own narrower default.
#
# Pick ONE of the two configs below depending on what you actually want
# to benchmark, and use the SAME one for the random baseline and your
# PPO agent so the comparison is apples-to-apples.

# Option A: match intersection-v0's real default (3 actions, speed only).
DISCRETE_META_ACTION_DEFAULT = {
    "type": "DiscreteMetaAction",
    "longitudinal": True,
    "lateral": False,
    "target_speeds": [0, 4.5, 9],
}

# Option B: full meta-action set as shown in the project PDF's example
# (5 actions: LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER). This is what
# {"type": "DiscreteMetaAction"} resolves to when longitudinal/lateral
# aren't explicitly restricted.
DISCRETE_META_ACTION_FULL = {
    "type": "DiscreteMetaAction",
}

# Active choice. Confirmed against the installed highway-env version:
#   >>> gym.make("intersection-v0").unwrapped.config["action"]
#   {'type': 'DiscreteMetaAction', 'longitudinal': True, 'lateral': False,
#    'target_speeds': [0, 4.5, 9]}
#   >>> env.action_space
#   Discrete(3)
# So the real env default is the 3-action speed-only version, not the
# PDF's bare 5-action example. Using DISCRETE_META_ACTION_DEFAULT so the
# random baseline matches whatever your PPO agent was actually trained
# against.
DISCRETE_META_ACTION = DISCRETE_META_ACTION_DEFAULT


# ---------------------------------------------------------------------
# Select which observation variant is active.
# Switch to "occupancy_grid" only if you intend to compare against the
# spatial/perception-style variant; the project's default benchmark uses
# "kinematic".
# ---------------------------------------------------------------------
OBSERVATION_VARIANT = "kinematic"

if OBSERVATION_VARIANT == "kinematic":
    observation_config = KINEMATIC_OBS
elif OBSERVATION_VARIANT == "occupancy_grid":
    observation_config = OCCUPANCY_GRID_OBS
else:
    raise ValueError(f"Unknown observation variant: {OBSERVATION_VARIANT}")


def get_env_config():
    """Return the full env config dict to pass to gym.make(..., config=...)."""
    return {
        "observation": observation_config,
        "action": DISCRETE_META_ACTION,
    }


ENV_ID = "intersection-v0"