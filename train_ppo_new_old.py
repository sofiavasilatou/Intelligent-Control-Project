"""
train_ppo.py
============
Train a PPO agent on intersection-v0.

Usage:  python train_ppo.py

Outputs:
    models/ppo_final.zip     <- use with PPOController / SafePPOController
    models/vecnormalize.pkl  <- observation normalizer stats (required at eval)
    models/ppo_best/         <- best checkpoint saved during training
    runs/ppo/                <- TensorBoard logs
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    BaseCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from env_config import make_env

# ── Hyperparameters ───────────────────────────────────────────────────────────
#
# KEY CHANGE: 500k steps minimum.  Intersection-v0 is a sparse, stochastic
# task.  80k steps (39 gradient updates) is nowhere near enough to learn
# any meaningful policy — the agent was essentially still random.
#
# Rule of thumb for highway-env tasks:
#   300k steps  → agent starts yielding reliably
#   500k steps  → solid policy, beats rule-based baseline
#   1M  steps   → further refinement, diminishing returns
#
TOTAL_TIMESTEPS  = 500_000

# Using 4 parallel envs multiplies effective experience by 4x with almost no
# extra wall-clock time on a modern CPU.  Each env runs independently with its
# own random seed, giving the agent much broader coverage of traffic scenarios.
N_ENVS           = 4

LEARNING_RATE    = 3e-4
# N_STEPS is per-env.  Total rollout per update = N_STEPS * N_ENVS = 8192.
# This is a healthy batch size for a discrete-action task.
N_STEPS          = 2048
BATCH_SIZE       = 256      # larger batch → more stable gradient estimates
N_EPOCHS         = 10       # more passes over the rollout data
GAMMA            = 0.99
GAE_LAMBDA       = 0.95
CLIP_RANGE       = 0.2
# Higher entropy coefficient keeps the policy exploratory longer, which is
# critical when rewards are sparse (most episodes end in few steps).
ENT_COEF         = 0.05

EVAL_FREQ        = 25_000   # evaluate every 25k steps (per env, so every ~6k real steps)
N_EVAL_EPISODES  = 20       # enough episodes for a statistically stable estimate

os.makedirs("models/checkpoints", exist_ok=True)
os.makedirs("models/ppo_best",    exist_ok=True)
os.makedirs("runs",               exist_ok=True)


# ── Normalizer sync callback ───────────────────────────────────────────────────
# FIX: The eval_env VecNormalize must track the SAME running statistics as the
# train_env, otherwise the best-model evaluation uses wrongly-scaled observations
# and the saved "best model" may actually be a worse checkpoint.
class SyncNormalizerCallback(BaseCallback):
    """Copy train_env normalizer stats into eval_env before each evaluation."""
    def __init__(self, train_env: VecNormalize, eval_env: VecNormalize, verbose=0):
        super().__init__(verbose)
        self.train_env = train_env
        self.eval_env  = eval_env

    def _on_step(self) -> bool:
        self.eval_env.obs_rms    = self.train_env.obs_rms
        self.eval_env.ret_rms    = self.train_env.ret_rms
        self.eval_env.clip_obs   = self.train_env.clip_obs
        self.eval_env.clip_reward = self.train_env.clip_reward
        return True


# ── Environments ──────────────────────────────────────────────────────────────
train_env = make_vec_env(make_env, n_envs=N_ENVS)
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)

eval_env = make_vec_env(make_env, n_envs=1)
eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

# ── Callbacks ─────────────────────────────────────────────────────────────────
sync_cb = SyncNormalizerCallback(train_env, eval_env)

checkpoint_cb = CheckpointCallback(
    save_freq=100_000 // N_ENVS,   # save every ~100k real steps
    save_path="models/checkpoints/",
    name_prefix="ppo_ckpt",
    verbose=1,
)

eval_cb = EvalCallback(
    eval_env,
    best_model_save_path="models/ppo_best/",
    log_path="runs/ppo_eval/",
    eval_freq=EVAL_FREQ // N_ENVS,  # EvalCallback counts per-env steps
    n_eval_episodes=N_EVAL_EPISODES,
    deterministic=True,
    verbose=1,
)

# ── Model ─────────────────────────────────────────────────────────────────────
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
    policy_kwargs={"net_arch": [256, 256]},
    verbose=1,
    tensorboard_log="runs/",
    device="auto",
)

# ── Train ─────────────────────────────────────────────────────────────────────
print(f"Training for {TOTAL_TIMESTEPS:,} steps across {N_ENVS} parallel envs...")
print(f"Effective updates: ~{TOTAL_TIMESTEPS // (N_STEPS * N_ENVS)} gradient steps")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[sync_cb, checkpoint_cb, eval_cb],
    tb_log_name="ppo",
    progress_bar=True,
)

# ── Save ──────────────────────────────────────────────────────────────────────
model.save("models/ppo_final")
train_env.save("models/vecnormalize.pkl")

print("\nDone.")
print("  Model saved to:       models/ppo_final.zip")
print("  Normalizer saved to:  models/vecnormalize.pkl")
print("  Best checkpoint at:   models/ppo_best/best_model.zip")
print("\nTIP: Use 'models/ppo_best/best_model.zip' in PPOController if")
print("     ppo_final.zip shows signs of overfitting on your eval runs.")

train_env.close()
eval_env.close()
