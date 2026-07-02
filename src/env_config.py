import gymnasium as gym
import highway_env  


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
    "flatten": False,               
    "observe_intentions": False,    
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
                "lateral": False,
                "target_speeds": [0, 4.5, 9],    
            },

            # Navigation/Routing
            "destination": "o1",                 
            "centering_position": [0.5, 0.6],    

            # Rewards
            "collision_reward": -5.0,
            "high_speed_reward": 1.0,
            "arrived_reward": 1.0,              
            "normalize_reward": False,

            # Simulation Settings
            "initial_vehicle_count": 10,
            "duration": 13,
            "controlled_vehicles": 1,
        },
    )
    return env