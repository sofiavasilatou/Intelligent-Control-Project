"""
train_ppo.py
============
Train a PPO agent on intersection-v1.

Usage:  python train_ppo.py

Outputs:
    models/ppo_final.zip   <- use this in PPOController / SafePPOController
    models/ppo_best/       <- best checkpoint
    runs/ppo/              <- TensorBoard logs
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from env_config import make_env

# ── Hyperparameters ──────────────────────────────────────────────────────────
# ── Optimised Hyperparameters ───────────────────────────────────────────────
# ── Hyperparameters ──────────────────────────────────────────────────────────
# ── Hyperparameters (Optimized for 5-minute Live Run) ────────────────────────
TOTAL_TIMESTEPS = 500_000  # Lowered step count: plenty for an easy 5-car setup to converge!
N_ENVS          = 8       
LEARNING_RATE   = 3e-4     
N_STEPS         = 2048     # Collects a strong chunk of 2,048 steps per update
BATCH_SIZE      = 256       
N_EPOCHS        = 10        # Halved epochs to cut policy update time in half!
GAMMA           = 0.99
GAE_LAMBDA      = 0.95
CLIP_RANGE      = 0.2
ENT_COEF        = 0.05     
EVAL_FREQ       = 20_000   # Only evaluates twice during the entire training run
N_EVAL_EPISODES = 30        # Evaluates over just 3 quick test episodes

os.makedirs("models/checkpoints", exist_ok=True)
os.makedirs("runs", exist_ok=True)

# ── Environments ─────────────────────────────────────────────────────────────
train_env = make_vec_env(make_env, n_envs=N_ENVS)
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)

eval_env = make_vec_env(make_env, n_envs=1)
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

# ── Callbacks ────────────────────────────────────────────────────────────────
checkpoint_cb = CheckpointCallback(
    save_freq=50_000,
    save_path="models/checkpoints/",
    name_prefix="ppo_ckpt",
    verbose=1,
)

eval_cb = EvalCallback(
    eval_env,
    best_model_save_path="models/ppo_best/",
    log_path="runs/ppo_eval/",
    eval_freq=EVAL_FREQ,
    n_eval_episodes=N_EVAL_EPISODES,
    deterministic=True,
    verbose=1,
)

# ── Model ────────────────────────────────────────────────────────────────────
model = PPO(
    policy="MlpPolicy",
    env=train_env,
    learning_rate=LEARNING_RATE,
    n_steps=N_STEPS,
    batch_size=BATCH_SIZE,
    n_epochs=N_EPOCHS,
    gamma=GAMMA,
    gae_lambda=GAE_LAMBDA,
    clip_range=CLIP_RANGE,
    ent_coef=ENT_COEF,
    policy_kwargs={"net_arch": [256, 256]},  # bigger network
    verbose=1,
    tensorboard_log="runs/",
    device="auto",
)

# ── Train ────────────────────────────────────────────────────────────────────
print(f"Training for {TOTAL_TIMESTEPS:,} steps across {N_ENVS} envs...")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[checkpoint_cb, eval_cb],
    tb_log_name="ppo",
    progress_bar=True,
)

# Save model AND the normalisation statistics (needed at eval time)
model.save("models/ppo_final")
train_env.save("models/vecnormalize.pkl")
print("\nDone.")
print("  Model saved to:       models/ppo_final.zip")
print("  Normalizer saved to:  models/vecnormalize.pkl")

train_env.close()
eval_env.close()