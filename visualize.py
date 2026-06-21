"""
visualize.py
============
Watch your trained controllers drive in intersection-v0.

Usage
-----
    # Watch PPO agent (default, 5 episodes):
    python visualize.py

    # Watch a specific controller:
    python visualize.py --controller random
    python visualize.py --controller rule_based
    python visualize.py --controller ppo
    python visualize.py --controller safe_ppo

    # Watch more episodes:
    python visualize.py --controller ppo --episodes 10

    # Use best checkpoint instead of final model:
    python visualize.py --model models/ppo_best/best_model.zip
"""

import argparse
import time
import numpy as np
from env_config import make_env


def watch(controller, env, n_episodes, delay=0.15):
    """Run n_episodes and render each step."""
    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        controller.reset(seed=ep)

        total_reward = 0.0
        steps        = 0
        crashed      = False

        print(f"\n--- Episode {ep + 1} ---")

        while True:
            action, act_info = controller.act(obs)

            # Print step info
            ctrl_name = act_info.get("controller", "?")
            shielded  = act_info.get("shielded", False)
            action_names = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT",
                            3: "FASTER",    4: "SLOWER"}
            shield_str = " [SHIELD]" if shielded else ""
            print(f"  step {steps:>3} | action: {action_names.get(action, action):<12}{shield_str}")

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps        += 1

            time.sleep(delay)   # slow down so you can watch

            if terminated or truncated:
                crashed = info.get("crashed", False)
                break

        status = "CRASHED" if crashed else "SUCCESS"
        print(f"  Result: {status} | reward: {total_reward:.2f} | steps: {steps}")

    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", default="ppo",
                        choices=["random", "rule_based", "ppo", "safe_ppo"])
    parser.add_argument("--model",    default="models/ppo_final.zip",
                        help="Path to PPO model zip (used for ppo / safe_ppo)")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--delay",    type=float, default=0.15,
                        help="Seconds between steps (increase to slow down)")
    args = parser.parse_args()

    # Build controller
    if args.controller == "random":
        from random_controller import RandomController
        ctrl = RandomController()

    elif args.controller == "rule_based":
        from rule_based_controller import RuleBasedController
        ctrl = RuleBasedController()

    elif args.controller == "ppo":
        from ppo_controller import PPOController
        ctrl = PPOController(args.model)

    elif args.controller == "safe_ppo":
        from safe_ppo_controller import SafePPOController
        ctrl = SafePPOController(args.model)

    # Make env with rendering enabled
    env = make_env(render_mode="human")

    print(f"\nWatching [{args.controller}] for {args.episodes} episodes...")
    print("Close the render window to stop early.\n")

    try:
        watch(ctrl, env, args.episodes, delay=args.delay)
    except KeyboardInterrupt:
        print("\nStopped by user.")
        env.close()


if __name__ == "__main__":
    main()