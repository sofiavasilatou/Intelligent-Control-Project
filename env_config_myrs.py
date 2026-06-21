from env_config import make_env
import numpy as np

env = make_env()
obs, info = env.reset(seed=0)
for _ in range(5):
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    print(np.array(obs))
    if term or trunc:
        break