"""
env_config.py
=============

Shared environment for training/evaluation
"""

import gymnasium as gym
import highway_env


KINEMATIC_OBS = {

    "type":"Kinematics",

    "vehicles_count":15,

    "features":[
        "presence",
        "x",
        "y",
        "vx",
        "vy",
        "cos_h",
        "sin_h"
    ],

    "features_range":{

        "x":[-100,100],
        "y":[-100,100],
        "vx":[-20,20],
        "vy":[-20,20]
    },

    "absolute":False,
    "order":"sorted"
}


def make_env(render_mode=None):

    env = gym.make(

        "intersection-v0",

        render_mode=render_mode,

        config={

            "observation":KINEMATIC_OBS,

            "action":{

                "type":"DiscreteMetaAction",

                "longitudinal":True,
                "lateral":False
            },

            # episode duration
            "duration":20,

            # traffic
            "controlled_vehicles":1,
            "initial_vehicle_count":8,
            "spawn_probability":0.6,

            # rewards
            "collision_reward":-5,
            "high_speed_reward":1,
            "arrived_reward":5,

            "reward_speed_range":[7.0,9.0],

            "normalize_reward":False,

            "offroad_terminal":False
        }
    )

    return env