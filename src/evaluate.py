
import argparse
import numpy as np

from env_config import make_env
from random_controller import RandomController
from rule_based_controller import RuleBasedController


def compute_ttc(obs):
    
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


def _episode_arrived(info):
    rewards = info.get("rewards", {})
    return int(rewards.get("arrived_reward", 0.0) >= 1.0)


def run_episodes(controller, n_episodes, base_seed):
    rewards, steps, crashes, arrivals, ttcs, shield_steps, total_steps = \
        [], [], [], [], [], [], []

    env = make_env()

    for ep in range(n_episodes):
        obs, info = env.reset(seed=base_seed + ep)
        controller.reset(seed=base_seed + ep)

        total_reward = 0.0
        ep_steps     = 0
        ep_min_ttc   = float("inf")
        ep_shield    = 0

        while True:
            action, act_info = controller.act(obs)

            ttc = compute_ttc(np.array(obs))
            ep_min_ttc = min(ep_min_ttc, ttc)
            if act_info.get("shielded", False):
                ep_shield += 1

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            ep_steps     += 1

            if terminated or truncated:
                break

        rewards.append(total_reward)
        steps.append(ep_steps)

        #   crashed  → info["crashed"] = True
        #   arrived  → info["rewards"]["arrived_reward"] = 1.0
        #   timeout  → truncated, neither crashed nor arrived
        crashed = int(info.get("crashed", False))
        arrived = _episode_arrived(info)

        crashes.append(crashed)
        arrivals.append(arrived)
        ttcs.append(ep_min_ttc if ep_min_ttc < float("inf") else np.nan)
        shield_steps.append(ep_shield)
        total_steps.append(ep_steps)

        if (ep + 1) % 20 == 0:
            print(f"    {ep+1}/{n_episodes} episodes done")

    env.close()

    crash_rate   = np.mean(crashes)
    arrival_rate = np.mean(arrivals)
    timeout_rate = 1.0 - crash_rate - arrival_rate

    return {
        "success_rate":   arrival_rate,
        "timeout_rate":   timeout_rate,
        "collision_rate": crash_rate,
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
                        help="Path to trained PPO model (without .zip)")
    parser.add_argument("--controllers", nargs="+",
                        default=["random", "rule_based", "ppo", "safe_ppo"],
                        choices=["random", "rule_based", "ppo", "safe_ppo"])
    args = parser.parse_args()

    to_eval = []
    for name in args.controllers:
        if name == "random":
            to_eval.append(("random", RandomController()))
        elif name == "rule_based":
            to_eval.append(("rule_based", RuleBasedController()))
        elif name == "ppo":
            if args.model is None:
                print("[SKIP] ppo — pass --model path/to/ppo_final")
                continue
            from ppo_controller_frame import PPOController
            to_eval.append(("ppo", PPOController(args.model)))
        elif name == "safe_ppo":
            if args.model is None:
                print("[SKIP] safe_ppo — pass --model path/to/ppo_final")
                continue
            from safe_ppo_controller_frame import SafePPOController
            to_eval.append(("safe_ppo", SafePPOController(args.model)))

    all_results = {}
    for name, ctrl in to_eval:
        print(f"\n{'='*50}")
        print(f"  Evaluating: {name}  ({args.episodes} episodes)")
        print(f"{'='*50}")
        all_results[name] = run_episodes(ctrl, args.episodes, args.seed)

    if not all_results:
        print("Nothing to evaluate.")
        return

    metrics = [
        "success_rate",    # actually crossed the intersection
        "timeout_rate",    # ran out of time
        "collision_rate",  # crashed
        "avg_reward",
        "std_reward",
        "avg_steps",
        "avg_min_ttc",
        "shield_rate",
    ]

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
    print()
    print("  NOTE: success_rate + timeout_rate + collision_rate = 1.0")


if __name__ == "__main__":
    main()