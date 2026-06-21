"""
debug_shield_threshold.py
==========================
Logs the closest-vehicle distance at every step across several episodes,
so we can pick a sensible `danger_dist` for SafePPOController empirically
instead of guessing.

Usage:
    python debug_shield_threshold.py --model models/ppo_final.zip --episodes 30
"""

import argparse
import numpy as np

from env_config import make_env
from ppo_controller import PPOController


def closest_vehicle_dist(observation):
    """Distance to the nearest other (present) vehicle, in observation units."""
    obs = np.array(observation)
    min_dist = float("inf")
    min_vx = None
    for i in range(1, obs.shape[0]):
        if obs[i, 0] < 0.5:  # not present
            continue
        x, y = obs[i, 1], obs[i, 2]
        vx = obs[i, 3]
        dist = np.sqrt(x**2 + y**2)
        if dist < min_dist:
            min_dist = dist
            min_vx = vx
    return min_dist, min_vx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    controller = PPOController(args.model)
    env = make_env()

    all_dists = []          # closest-vehicle distance at every step, every episode
    crash_step_dists = []   # closest-vehicle distance on the step BEFORE a crash
    safe_ep_min_dists = []  # min distance seen in episodes that did NOT crash
    crash_ep_min_dists = [] # min distance seen in episodes that DID crash

    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        controller.reset(seed=args.seed + ep)

        ep_dists = []
        prev_dist = None
        crashed = False

        while True:
            dist, vx = closest_vehicle_dist(obs)
            ep_dists.append(dist)
            all_dists.append(dist)

            action, _ = controller.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated and info.get("crashed", False):
                crashed = True
                if prev_dist is not None:
                    crash_step_dists.append(prev_dist)
                else:
                    crash_step_dists.append(dist)

            prev_dist = dist

            if terminated or truncated:
                break

        finite_ep_dists = [d for d in ep_dists if d < float("inf")]
        ep_min = min(finite_ep_dists) if finite_ep_dists else float("nan")

        if crashed:
            crash_ep_min_dists.append(ep_min)
        else:
            safe_ep_min_dists.append(ep_min)

    env.close()

    finite_all = np.array([d for d in all_dists if d < float("inf")])

    print("\n" + "=" * 60)
    print("  CLOSEST-VEHICLE DISTANCE DIAGNOSTICS")
    print("=" * 60)
    print(f"  Episodes run:              {args.episodes}")
    print(f"  Crashed episodes:          {len(crash_ep_min_dists)}")
    print(f"  Safe episodes:             {len(safe_ep_min_dists)}")
    print("-" * 60)
    print(f"  Overall step distances     n={len(finite_all)}")
    print(f"    min:    {finite_all.min():.4f}")
    print(f"    p10:    {np.percentile(finite_all, 10):.4f}")
    print(f"    p25:    {np.percentile(finite_all, 25):.4f}")
    print(f"    median: {np.median(finite_all):.4f}")
    print(f"    mean:   {finite_all.mean():.4f}")
    print("-" * 60)
    if crash_step_dists:
        cs = np.array(crash_step_dists)
        print(f"  Distance on step BEFORE a crash (n={len(cs)}):")
        print(f"    min:    {cs.min():.4f}")
        print(f"    median: {np.median(cs):.4f}")
        print(f"    max:    {cs.max():.4f}")
    else:
        print("  No crashes observed in this sample.")
    print("-" * 60)
    if crash_ep_min_dists:
        cmin = np.array(crash_ep_min_dists)
        print(f"  Min distance reached in CRASHED episodes (n={len(cmin)}):")
        print(f"    min:    {cmin.min():.4f}")
        print(f"    median: {np.median(cmin):.4f}")
        print(f"    max:    {cmin.max():.4f}")
    if safe_ep_min_dists:
        smin = np.array(safe_ep_min_dists)
        print(f"  Min distance reached in SAFE episodes (n={len(smin)}):")
        print(f"    min:    {smin.min():.4f}")
        print(f"    median: {np.median(smin):.4f}")
        print(f"    max:    {smin.max():.4f}")
    print("=" * 60)

    print("\nSuggested danger_dist candidates to try:")
    if crash_step_dists:
        suggestion = np.percentile(crash_step_dists, 75)
        print(f"  ~{suggestion:.2f}  (75th percentile of pre-crash distances —"
              f" catches most close calls with some margin)")
    print(f"  ~{np.percentile(finite_all, 25):.2f}  (25th percentile of all step distances —"
          f" a more conservative/frequent trigger)")


if __name__ == "__main__":
    main()