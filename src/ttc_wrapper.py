import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

class TTCObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_space = self.observation_space
        low = obs_space.low
        high = obs_space.high
        
        new_low = np.hstack([low, np.zeros((low.shape[0], 1))])
        new_high = np.hstack([high, np.full((high.shape[0], 1), 10.0)])
        
        self.observation_space = Box(low=new_low, high=new_high, dtype=np.float32)

    def observation(self, obs):
        ttcs = np.zeros((obs.shape[0], 1), dtype=np.float32)
        ego_vx, ego_vy = obs[0, 3], obs[0, 4]
        
        for i in range(1, obs.shape[0]):
            if obs[i, 0] < 0.5:
                ttcs[i, 0] = 10.0
                continue
                
            x, y = obs[i, 1], obs[i, 2]
            vx, vy = obs[i, 3], obs[i, 4]
            dist = np.sqrt(x**2 + y**2)
            
            if dist < 1e-5:
                ttcs[i, 0] = 10.0
                continue
                
            dx, dy = x / dist, y / dist
            closing_speed = -((vx - ego_vx) * dx + (vy - ego_vy) * dy)
            
            if closing_speed > 0.01:
                ttcs[i, 0] = min(dist / closing_speed, 10.0)
            else:
                ttcs[i, 0] = 10.0
                
        return np.hstack([obs, ttcs], dtype=np.float32)

class TTCRewardWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        min_ttc = 10.0
        ego_vx, ego_vy = obs[0, 3], obs[0, 4]
        
        for i in range(1, obs.shape[0]):
            if obs[i, 0] > 0.5:
                x, y = obs[i, 1], obs[i, 2]
                vx, vy = obs[i, 3], obs[i, 4]
                dist = np.sqrt(x**2 + y**2)
                
                if dist > 1e-5:
                    dx, dy = x / dist, y / dist
                    closing_speed = -((vx - ego_vx) * dx + (vy - ego_vy) * dy)
                    
                    if closing_speed > 0.01:
                        min_ttc = min(min_ttc, dist / closing_speed)
        
        if min_ttc < 2.0:
            reward -= (2.0 - min_ttc) * 1.0  
            
        return obs, reward, terminated, truncated, info