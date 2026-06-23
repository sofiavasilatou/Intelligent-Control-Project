"""
Project 4: Safe Autonomous Driving at Intersections
Generic evaluation harness: runs N episodes with a given policy function
and prints summary statistics to the console.
"""

import argparse

import gymnasium as gym
import highway_env  # noqa: F401  (registers the env)
import numpy as np

from env_config_myrs import ENV_ID, get_env_config
# IMPORT YOUR CONTROLLER HERE
from rule_based_controller import RuleBasedController


def make_env():
    """Build intersection-v0 using the shared project config."""
    return gym.make(
        ENV_ID,
        render_mode=None,
        config=get_env_config(),
    )


def random_policy(observation, env):
    """Uniform random policy: ignores the observation entirely."""
    return env.action_space.sample()


def make_rule_based_policy():
    """
    Creates and returns a policy function that wraps your RuleBasedController.
    This bridges the gap between your controller's act() method and the 
    policy_fn() signature expected by the evaluation loop.
    """
    # Instantiate the controller once
    controller = RuleBasedController()
    
    def policy_fn(observation, env):
        # Your act() method returns a tuple: (action, info)
        # We only need the action for the environment step.
        action, info = controller.act(observation)
        return action
        
    return policy_fn


def run_episode(env, policy_fn, seed):
    """Run a single episode with the given policy function."""
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

    # success = the episode ended via `terminated` and no crash was ever flagged
    success = terminated and not crashed

    return {
        "reward": episode_reward,
        "steps": steps,
        "crashed": crashed,
        "success": success,
    }


def evaluate(policy_fn, episodes=100, seed=0, label="policy", verbose=True):
    """Run `episodes` evaluation episodes and return a summary dict."""
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

    # Generate the rule based policy function
    rule_based_fn = make_rule_based_policy()
    
    # Run the evaluation!
    evaluate(rule_based_fn, episodes=args.episodes, seed=args.seed, label="rule_based")


if __name__ == "__main__":
    main()