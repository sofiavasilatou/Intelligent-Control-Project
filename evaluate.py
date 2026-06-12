"""
evaluate.py
===========
Run all controllers and print a comparison table.

Usage
-----
    # Baselines only (no trained model needed):
    python evaluate.py --controllers random rule_based

    # All controllers:
    python evaluate.py --model models/ppo_final.zip

    # Quick smoke test:
    python evaluate.py --episodes 5 --controllers random
"""

import argparse
import numpy as np

from env_config import make_env
from random_controller import RandomController
from rule_based_controller import RuleBasedController


def compute_ttc(obs):
    """Minimum time-to-collision from a kinematic observation row."""
    min_ttc = float("inf")
    for i in range(1, obs.shape[0]):
        if obs[i, 0] < 0.5:
            continue
        x, y, vx = obs[i, 1], obs[i, 2], obs[i, 3]
        dist = np.sqrt(x**2 + y**2)
        closing = -vx
        if closing > 0.01:
            min_ttc = min(min_ttc, dist / closing)
    return min_ttc


def run_episodes(controller, n_episodes, base_seed):
    rewards, steps, crashes, ttcs, shield_steps, total_steps = [], [], [], [], [], []

    env = make_env()

    for ep in range(n_episodes):
        obs, info = env.reset(seed=base_seed + ep)
        controller.reset(seed=base_seed + ep)

        total_reward = 0.0
        ep_steps = 0
        ep_min_ttc = float("inf")
        ep_shield = 0

        while True:
            action, act_info = controller.act(obs)

            ttc = compute_ttc(np.array(obs))
            ep_min_ttc = min(ep_min_ttc, ttc)
            if act_info.get("shielded", False):
                ep_shield += 1

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            ep_steps += 1

            if terminated or truncated:
                break

        rewards.append(total_reward)
        steps.append(ep_steps)
        crashes.append(int(info.get("crashed", False)))
        ttcs.append(ep_min_ttc if ep_min_ttc < float("inf") else np.nan)
        shield_steps.append(ep_shield)
        total_steps.append(ep_steps)

        if (ep + 1) % 20 == 0:
            print(f"    {ep+1}/{n_episodes} episodes done")

    env.close()

    return {
        "success_rate":   1 - np.mean(crashes),
        "collision_rate": np.mean(crashes),
        "avg_reward":     np.mean(rewards),
        "std_reward":     np.std(rewards),
        "avg_steps":      np.mean(steps),
        "avg_min_ttc":    np.nanmean(ttcs),
        "shield_rate":    sum(shield_steps) / max(sum(total_steps), 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes",    type=int, default=100)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--model",       type=str, default=None,
                        help="Path to trained PPO .zip file")
    parser.add_argument("--controllers", nargs="+",
                        default=["random", "rule_based", "ppo", "safe_ppo"],
                        choices=["random", "rule_based", "ppo", "safe_ppo"])
    args = parser.parse_args()

    # Build the list of (name, controller) to evaluate
    to_eval = []
    for name in args.controllers:
        if name == "random":
            to_eval.append(("random", RandomController()))
        elif name == "rule_based":
            to_eval.append(("rule_based", RuleBasedController()))
        elif name == "ppo":
            if args.model is None:
                print("[SKIP] ppo — pass --model path/to/ppo_final.zip")
                continue
            from ppo_controller import PPOController
            to_eval.append(("ppo", PPOController(args.model)))
        elif name == "safe_ppo":
            if args.model is None:
                print("[SKIP] safe_ppo — pass --model path/to/ppo_final.zip")
                continue
            from safe_ppo_controller import SafePPOController
            to_eval.append(("safe_ppo", SafePPOController(args.model)))

    # Run evaluation
    all_results = {}
    for name, ctrl in to_eval:
        print(f"\n{'='*50}")
        print(f"  Evaluating: {name}  ({args.episodes} episodes)")
        print(f"{'='*50}")
        all_results[name] = run_episodes(ctrl, args.episodes, args.seed)

    # Print comparison table
    if not all_results:
        print("Nothing to evaluate.")
        return

    metrics = ["success_rate", "collision_rate", "avg_reward",
               "std_reward", "avg_steps", "avg_min_ttc", "shield_rate"]

    col_w = 14
    print("\n" + "=" * (25 + col_w * len(all_results)))
    print("  COMPARISON TABLE")
    print("=" * (25 + col_w * len(all_results)))
    header = f"  {'metric':<23}" + "".join(f"{n:<{col_w}}" for n in all_results)
    print(header)
    print("-" * (25 + col_w * len(all_results)))
    for m in metrics:
        row = f"  {m:<23}"
        for stats in all_results.values():
            row += f"{stats[m]:<{col_w}.4f}"
        print(row)
    print("=" * (25 + col_w * len(all_results)))


if __name__ == "__main__":
    main()
