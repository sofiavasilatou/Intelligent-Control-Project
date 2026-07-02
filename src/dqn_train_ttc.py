"""
dqn_train_myrs.py
==================
Project 4: Safe Autonomous Driving at Intersections
DQN training via Stable-Baselines3 with TTC-augmented observations.
"""

import argparse
import os

import gymnasium as gym
import highway_env  # noqa: F401
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

# Assume these are defined in your local files
from ttc_wrapper import TTCObsWrapper, TTCRewardWrapper
from progress_callback import ProgressCallback

# ---------------------------------------------------------
# 1. Aligned Environment Configuration
# ---------------------------------------------------------
KINEMATIC_OBS = {
    "type": "Kinematics",
    "vehicles_count": 15,
    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
    "features_range": {
        "x": [-100, 100],
        "y": [-100, 100],
        "vx": [-20, 20],
        "vy": [-20, 20],
    },
    "absolute": False,               
    "flatten": False,               
    "observe_intentions": False,    
    "order": "sorted",               # ADDED: Required by benchmark to prevent dynamic shape crashes
}

BASE_ENV_CONFIG = {
    "observation": KINEMATIC_OBS,
    "action": {
        "type": "DiscreteMetaAction",
        "longitudinal": True,
        "lateral": False,
        "target_speeds": [0, 4.5, 9],
    },
    "destination": "o1",
    "centering_position": [0.5, 0.6],
    "collision_reward": -5.0,
    "high_speed_reward": 1.0,
    "arrived_reward": 1.0,
    "normalize_reward": False,
    "initial_vehicle_count": 10,
    "duration": 13,
    "controlled_vehicles": 1,
}

# ---------------------------------------------------------
# 2. Environment Factories
# ---------------------------------------------------------
def make_train_env_fn(seed: int = 0):
    def _init():
        # CRITICAL: Isolate random seeds for parallel workers
        set_random_seed(seed)
        env = gym.make("intersection-v0", render_mode=None, config=BASE_ENV_CONFIG)
        env = TTCObsWrapper(env)       # augment obs with TTC feature
        env = TTCRewardWrapper(env)    # add TTC shaping penalty
        return env
    return _init

def make_eval_env_fn(seed: int = 0):
    """Eval env: original rewards + TTC obs (same obs space as training)."""
    def _init():
        set_random_seed(seed)
        env = gym.make("intersection-v0", render_mode=None, config=BASE_ENV_CONFIG)
        env = TTCObsWrapper(env)   # must match training obs space (Warning: apply internally for final eval)
        return env
    return _init

# ---------------------------------------------------------
# 3. Main Training Loop
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--n_envs",    type=int, default=8)
    parser.add_argument("--out",       type=str, default="models_myrs/dqn_final")
    args = parser.parse_args()

    out_dir = os.path.dirname(args.out) if os.path.dirname(args.out) else "."
    os.makedirs(out_dir, exist_ok=True)

    train_env = SubprocVecEnv(
        [make_train_env_fn(seed=args.seed + i) for i in range(args.n_envs)]
    )
    train_env = VecMonitor(train_env)

    eval_env = SubprocVecEnv([make_eval_env_fn(seed=999)])
    eval_env = VecMonitor(eval_env)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=out_dir,
        log_path=out_dir,
        eval_freq=max(10_000 // args.n_envs, 1),
        n_eval_episodes=30,
        deterministic=True,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // args.n_envs, 1),
        save_path=out_dir,
        name_prefix="dqn_checkpoint",
        verbose=0,
    )
    progress_callback = ProgressCallback(
        total_timesteps=args.timesteps,
        print_freq=10_000,
    )

    # Flatten obs for DQN's MlpPolicy: (N, F+1) -> (N*(F+1),)
    # SB3's MlpPolicy auto-flattens Box observations natively.
    model = DQN(
        "MlpPolicy",
        train_env,
        policy_kwargs=dict(net_arch=[512, 512]),
        learning_rate=1e-4,
        buffer_size=200_000,
        learning_starts=10_000,
        batch_size=256,
        gamma=0.98,
        train_freq=4,
        gradient_steps=2,
        target_update_interval=2_000,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        max_grad_norm=10.0,
        seed=args.seed,
        verbose=1,
    )

    print(f"Training DQN with TTC-augmented obs for {args.timesteps} steps "
          f"across {args.n_envs} parallel envs...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=[eval_callback, checkpoint_callback, progress_callback],
    )

    model.save(args.out)
    print(f"\nFinal model -> {args.out}.zip")
    print(f"Best model  -> {out_dir}/best_model.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()