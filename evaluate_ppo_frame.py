"""
evaluate_ppo.py
===============
Evaluate the trained PPO agent on intersection-v0.

This script uses your PPOController, which automatically handles the
observation normalization and frame stacking required by the model.
"""

import argparse
import gymnasium as gym
import highway_env  # Registers the env
import numpy as np

# Adjust this import to match your environment config file name
# (e.g., env_config or env_config_myrs)
from env_config import make_env, KINEMATIC_OBS

from ppo_controller_frame import PPOController

# Match this to the environment ID used in your training
ENV_ID = "intersection-v0"


def make_ppo_policy(model_path, normalizer_path):
    """
    Creates and returns a policy function that wraps your PPOController.
    """
    # Initialize the controller with the paths to your trained files
    controller = PPOController(
        model_path=model_path,
        normalizer_path=normalizer_path,
        deterministic=True # Use greedy actions for evaluation
    )
    
    def policy_fn(observation, env):
        # Your act() method returns a tuple: (action, info)
        action, info = controller.act(observation)
        return action
        
    return policy_fn, controller


def run_episode(env, policy_fn, controller, seed):
    """Run a single episode with the given policy function."""
    obs, info = env.reset(seed=seed)
    
    # CRITICAL: Reset the controller's frame stack at the start of each episode
    controller.reset(seed=seed)
    
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


def evaluate(policy_fn, controller, episodes=100, seed=0, label="PPO", verbose=True):
    """Run `episodes` evaluation episodes and return a summary dict."""
    env = make_env()

    rewards = np.zeros(episodes)
    lengths = np.zeros(episodes, dtype=int)
    crashes = np.zeros(episodes, dtype=bool)
    successes = np.zeros(episodes, dtype=bool)

    for ep in range(episodes):
        result = run_episode(env, policy_fn, controller, seed=seed + ep)
        rewards[ep] = result["reward"]
        lengths[ep] = result["steps"]
        crashes[ep] = result["crashed"]
        successes[ep] = result["success"]

        if verbose:
            print(
                f"[{label} | ep {ep:03d}] "
                f"reward={result['reward']:>6.2f} "
                f"steps={result['steps']:>3d} "
                f"crashed={result['crashed']:<5} "
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
        print(f"Reward        : {summary['mean_reward']:.3f} +/- {summary['std_reward']:.3f}")
        print(f"Episode length: {summary['mean_length']:.1f} +/- {summary['std_length']:.1f} steps")
        print(f"Crash rate    : {summary['crash_rate']*100:.1f}%")
        print(f"Success rate  : {summary['success_rate']*100:.1f}%")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained PPO on intersection-v0")
    parser.add_argument("--episodes", type=int, default=100,
                         help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=0,
                         help="Base seed; episode i uses seed + i")
    
    # Add arguments so you can easily swap between your "final" model and your "best" model via command line
    parser.add_argument("--model", type=str, default="models_myrs_env/ppo_best/best_model.zip",
                         help="Path to the trained model zip file")
    parser.add_argument("--normalizer", type=str, default="models_myrs_env/vecnormalize.pkl",
                         help="Path to the saved VecNormalize statistics")
    args = parser.parse_args()

    print(f"Loading Model: {args.model}")
    print(f"Loading Norm : {args.normalizer}")
    print("-" * 50)

    # Generate the PPO policy function and retrieve the controller instance
    ppo_fn, ppo_controller = make_ppo_policy(
        model_path=args.model, 
        normalizer_path=args.normalizer
    )
    
    # Run the evaluation!
    evaluate(ppo_fn, ppo_controller, episodes=args.episodes, seed=args.seed, label="PPO")


if __name__ == "__main__":
    main()