"""
quickstart.py
=============
Verifies the setup works. Run this first — no training needed.

Usage
-----
    python quickstart.py
"""

import numpy as np
from env_config import make_env
from random_controller import RandomController
from rule_based_controller import RuleBasedController


def run_episode(controller, seed=0):
    env = make_env()
    obs, info = env.reset(seed=seed)
    controller.reset(seed=seed)

    total_reward = 0.0
    steps = 0

    while True:
        action, _ = controller.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break

    crashed = info.get("crashed", False)
    env.close()
    return total_reward, steps, crashed


if __name__ == "__main__":
    print("=" * 50)
    print("  Project 4 — Quick Start Sanity Check")
    print("=" * 50)

    for name, ctrl in [("Random", RandomController()), ("Rule-Based", RuleBasedController())]:
        reward, steps, crashed = run_episode(ctrl, seed=42)
        status = "CRASHED" if crashed else "OK"
        print(f"  {name:<15} reward={reward:7.2f}  steps={steps:4d}  {status}")

    print("\nAll good! Next steps:")
    print("  pip install -r requirements.txt")
    print("  python train_ppo.py")
    print("  python evaluate.py --model models/ppo_final.zip")
