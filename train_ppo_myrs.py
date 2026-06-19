"""
train_ppo.py
============

Train PPO for intersection-v0
"""

import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecNormalize
)
from stable_baselines3.common.callbacks import EvalCallback

from env_config_myrs import make_env


def main():

    train_env = DummyVecEnv(
        [make_env]
    )

    train_env = VecNormalize(

        train_env,

        norm_obs=True,
        norm_reward=True,
        clip_obs=10
    )


    eval_env = DummyVecEnv(
        [make_env]
    )

    eval_env = VecNormalize(

        eval_env,

        norm_obs=True,
        norm_reward=False,

        training=False
    )


    model = PPO(

        "MlpPolicy",

        train_env,

        learning_rate=3e-4,

        n_steps=2048,

        batch_size=64,

        gamma=0.99,

        gae_lambda=0.95,

        clip_range=0.2,

        ent_coef=0.01,

        verbose=1,

        tensorboard_log="./logs/"
    )


    eval_callback = EvalCallback(

        eval_env,

        best_model_save_path="./models/",

        log_path="./logs/",

        eval_freq=5000,

        deterministic=True
    )


    os.makedirs(
        "models",
        exist_ok=True
    )


    print("\nTraining PPO...\n")

    model.learn(

        total_timesteps=300000,

        callback=eval_callback
    )


    model.save(
        "models/ppo_final"
    )

    train_env.save(
        "models/vecnormalize.pkl"
    )


    print("\nTraining complete")
    print("Saved model → models/ppo_final.zip")


if __name__=="__main__":
    main()