import argparse
import json

import numpy as np

from env_config import make_env


class RandomController:

    def __init__(self, n_actions: int = None):
        self._n_actions = n_actions
        self._rng = np.random.default_rng()

    def reset(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)

    def act(self, obs):
        if self._n_actions is None:
            _env = make_env()
            self._n_actions = _env.action_space.n
            _env.close()
        action = int(self._rng.integers(0, self._n_actions))
        return action, {"shielded": False}


def run_episode(env, controller, seed: int):
    obs, info = env.reset(seed=seed)
    controller.reset(seed=seed)

    episode_reward = 0.0
    steps          = 0
    ep_crashed     = False
    ep_arrived     = False

    while True:
        action, _ = controller.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        steps += 1

        if info.get("crashed", False):
            ep_crashed = True

        # Mirror evaluate.py: latch arrival from per-step rewards dict
        if info.get("rewards", {}).get("arrived_reward", 0.0) >= 1.0:
            ep_arrived = True

        if terminated or truncated:
            break

    return {
        "reward":  episode_reward,
        "steps":   steps,
        "crashed": ep_crashed,
        "success": ep_arrived and not ep_crashed,
    }


def main():
    parser = argparse.ArgumentParser(description="Random baseline for intersection-v0")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed",     type=int, default=0)
    parser.add_argument("--out",      type=str, default="random_myrs_results.json")
    args = parser.parse_args()

    env        = make_env()
    controller = RandomController(n_actions=env.action_space.n)

    results = []
    for ep in range(args.episodes):
        ep_result = run_episode(env, controller, seed=args.seed + ep)
        ep_result["episode"] = ep
        results.append(ep_result)
        print(
            f"[ep {ep:03d}] reward={ep_result['reward']:.2f} "
            f"steps={ep_result['steps']} "
            f"crashed={ep_result['crashed']} "
            f"success={ep_result['success']}"
        )

    env.close()

    rewards   = np.array([r["reward"]  for r in results])
    crashes   = np.array([r["crashed"] for r in results], dtype=float)
    successes = np.array([r["success"] for r in results], dtype=float)
    timeouts  = 1.0 - crashes - successes

    summary = {
        "episodes":     args.episodes,
        "mean_reward":  float(rewards.mean()),
        "std_reward":   float(rewards.std()),
        "crash_rate":   float(crashes.mean()),
        "success_rate": float(successes.mean()),
        "timeout_rate": float(timeouts.mean()),
    }

    print("\n=== Random baseline summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\n  CHECK: success + timeout + crash = "
          f"{summary['success_rate'] + summary['timeout_rate'] + summary['crash_rate']:.4f}")

    with open(args.out, "w") as f:
        json.dump({"summary": summary, "episodes": results}, f, indent=2)
    print(f"Saved results to {args.out}")


if __name__ == "__main__":
    main()