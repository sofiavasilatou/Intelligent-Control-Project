"""
Project 4: Safe Autonomous Driving at Intersections
Generic evaluation harness: runs N episodes with a given policy function
and prints summary statistics to the console.

Designed to be policy-agnostic: today it evaluates the uniform random
policy, but the same `evaluate()` function works for your PPO agent (or
any other controller) later — just pass in a different `policy_fn`.

A policy_fn has the signature:
    action = policy_fn(observation)

Run (random policy, default):
    python evaluate.py --episodes 100 --seed 0

The script prints per-episode lines as it goes, then a summary block at
the end with mean/std for reward and episode length, plus crash and
success rates. No files are written; this is for quick console checks
during development. Use random_baseline.py if you want the JSON dump
for the report.
"""

import argparse

import gymnasium as gym
import highway_env  # noqa: F401  (registers the env)
import numpy as np

from env_config_myrs import ENV_ID, get_env_config


def make_env():
    """Build intersection-v0 using the shared project config, so every
    policy evaluated with this script sees the exact same observation/
    action settings."""
    return gym.make(
        ENV_ID,
        render_mode=None,
        config=get_env_config(),
    )


def random_policy(observation, env):
    """Uniform random policy: ignores the observation entirely."""
    return env.action_space.sample()


def run_episode(env, policy_fn, seed):
    """Run a single episode with the given policy function.

    policy_fn is called as policy_fn(observation, env) so it has access
    to env.action_space if needed (useful for the random policy, and
    harmless for trained policies that ignore the env argument).
    """
    obs, info = env.reset(seed=seed)
    terminated, truncated = False, False

    episode_reward = 0.0
    steps = 0
    crashed = False

    while not (terminated or truncated):
        action = policy_fn(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        steps += 1

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


def evaluate(policy_fn, episodes=100, seed=0, label="policy", verbose=True):
    """Run `episodes` evaluation episodes and return a summary dict.

    Prints per-episode and summary stats to the console if verbose=True.
    """
    env = make_env()

    rewards = np.zeros(episodes)
    lengths = np.zeros(episodes, dtype=int)
    crashes = np.zeros(episodes, dtype=bool)
    successes = np.zeros(episodes, dtype=bool)

    for ep in range(episodes):
        result = run_episode(env, policy_fn, seed=seed + ep)
        rewards[ep] = result["reward"]
        lengths[ep] = result["steps"]
        crashes[ep] = result["crashed"]
        successes[ep] = result["success"]

        if verbose:
            print(
                f"[{label} | ep {ep:03d}] "
                f"reward={result['reward']:.2f} "
                f"steps={result['steps']} "
                f"crashed={result['crashed']} "
                f"success={result['success']}"
            )

    env.close()

    summary = {
        "label": label,
        "episodes": episodes,
        "mean_reward": float(rewards.mean()),
        "std_reward": float(rewards.std()),
        "mean_length": float(lengths.mean()),
        "std_length": float(lengths.std()),
        "crash_rate": float(crashes.mean()),
        "success_rate": float(successes.mean()),
    }

    if verbose:
        print(f"\n=== {label}: evaluation summary over {episodes} episodes ===")
        print(f"Reward       : {summary['mean_reward']:.3f} +/- {summary['std_reward']:.3f}")
        print(f"Episode length: {summary['mean_length']:.1f} +/- {summary['std_length']:.1f} steps")
        print(f"Crash rate   : {summary['crash_rate']*100:.1f}%")
        print(f"Success rate : {summary['success_rate']*100:.1f}%")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate a controller on intersection-v0")
    parser.add_argument("--episodes", type=int, default=100,
                         help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=0,
                         help="Base seed; episode i uses seed + i")
    args = parser.parse_args()

    # Currently wired to the random policy. To evaluate a trained model
    # later, define a new policy_fn, e.g.:
    #
    #   from stable_baselines3 import PPO
    #   model = PPO.load("ppo_intersection.zip")
    #
    #   def ppo_policy(observation, env):
    #       action, _ = model.predict(observation, deterministic=True)
    #       return action
    #
    # and call evaluate(ppo_policy, episodes=..., seed=..., label="PPO").
    evaluate(random_policy, episodes=args.episodes, seed=args.seed, label="random")


if __name__ == "__main__":
    main()