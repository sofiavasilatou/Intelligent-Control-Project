"""
Project 4: Safe Autonomous Driving at Intersections
Simple baseline: uniform random policy over the discrete meta-action space.

This is the required "simple baseline" (random policy) to compare against
the competing method (e.g. PPO) in the final report. It applies no learning
and no hand-crafted logic, so any improvement shown by the trained agent is
attributable to learning rather than to the environment being trivially easy.

Run:
    python random_myrs.py --episodes 100 --seed 0
"""

import argparse
import json

import gymnasium as gym
import highway_env  # noqa: F401  (registers the env)
import numpy as np

from env_config_myrs import ENV_ID, get_env_config


def make_env():
    """Build intersection-v0 using the shared project config, so this
    baseline and any other controller (e.g. PPO) are evaluated under the
    exact same observation/action settings."""
    env = gym.make(
        ENV_ID,
        render_mode=None,
        config=get_env_config(),
    )
    return env


def run_episode(env, seed):
    """Run a single episode with a uniform random policy.

    Returns a dict of per-episode metrics useful for the report's
    comparison table (reward, crash, success, length).
    """
    obs, info = env.reset(seed=seed)
    terminated, truncated = False, False

    episode_reward = 0.0
    steps = 0
    crashed = False

    while not (terminated or truncated):
        # Uniform random action: no learning, no rules. This is the
        # "simple baseline" required by the project spec.
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        steps += 1

        # HighwayEnv's info dict reports a "crashed" flag.
        if info.get("crashed", False):
            crashed = True

    # intersection-v0 terminates (not truncates) the episode either on a
    # collision or when the vehicle has successfully crossed the
    # intersection (has_arrived). Truncation only happens on a timeout,
    # i.e. the vehicle neither crashed nor arrived in time — that is a
    # failure to make progress, NOT a success.
    #
    # So: success = the episode ended via `terminated` and no crash was
    # ever flagged (i.e. it ended because the vehicle arrived).
    success = terminated and not crashed

    return {
        "reward": episode_reward,
        "steps": steps,
        "crashed": crashed,
        "success": success,
    }


def main():
    parser = argparse.ArgumentParser(description="Random baseline for intersection-v0")
    parser.add_argument("--episodes", type=int, default=100,
                         help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=0,
                         help="Base seed; episode i uses seed + i")
    parser.add_argument("--out", type=str, default="random_myrs_results.json",
                         help="Where to dump per-episode + summary results")
    args = parser.parse_args()

    env = make_env()

    results = []
    for ep in range(args.episodes):
        ep_result = run_episode(env, seed=args.seed + ep)
        ep_result["episode"] = ep
        results.append(ep_result)
        print(
            f"[ep {ep:03d}] reward={ep_result['reward']:.2f} "
            f"steps={ep_result['steps']} "
            f"crashed={ep_result['crashed']} "
            f"success={ep_result['success']}"
        )

    env.close()

    rewards = np.array([r["reward"] for r in results])
    crashes = np.array([r["crashed"] for r in results])
    successes = np.array([r["success"] for r in results])

    summary = {
        "episodes": args.episodes,
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
        "crash_rate": float(crashes.mean()),
        "success_rate": float(successes.mean()),
    }

    print("\n=== Random baseline summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    with open(args.out, "w") as f:
        json.dump({"summary": summary, "episodes": results}, f, indent=2)
    print(f"\nSaved full results to {args.out}")


if __name__ == "__main__":
    main()