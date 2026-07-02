import os
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from env_config import make_env


TOTAL_TIMESTEPS = 500_000

N_ENVS          = 1
LEARNING_RATE   = 1e-4
BUFFER_SIZE     = 50_000
LEARNING_STARTS = 1_000
BATCH_SIZE      = 64
TARGET_UPDATE_INTERVAL = 1_000
GAMMA           = 0.99


EXPLORATION_FRACTION   = 0.3   
EXPLORATION_FINAL_EPS  = 0.05   

EVAL_FREQ       = 10_000
N_EVAL_EPISODES = 30

os.makedirs("models/dqn_best",    exist_ok=True)
os.makedirs("models/checkpoints", exist_ok=True)
os.makedirs("runs",               exist_ok=True)


# Environments 
train_env = make_vec_env(make_env, n_envs=N_ENVS, vec_env_cls=DummyVecEnv)
eval_env  = make_vec_env(make_env, n_envs=1,      vec_env_cls=DummyVecEnv)


# Callbacks 
checkpoint_cb = CheckpointCallback(
    save_freq=50_000,
    save_path="models/checkpoints/",
    name_prefix="dqn_ckpt",
    verbose=1,
)

eval_cb = EvalCallback(
    eval_env,
    best_model_save_path="models/dqn_best/",
    log_path="runs/dqn_eval/",
    eval_freq=EVAL_FREQ,
    n_eval_episodes=N_EVAL_EPISODES,
    deterministic=True,
    verbose=1,
)


#  Model 
model = DQN(
    policy="MlpPolicy",
    env=train_env,
    learning_rate=LEARNING_RATE,
    buffer_size=BUFFER_SIZE,
    learning_starts=LEARNING_STARTS,
    batch_size=BATCH_SIZE,
    gamma=GAMMA,
    target_update_interval=TARGET_UPDATE_INTERVAL,
    exploration_fraction=EXPLORATION_FRACTION,
    exploration_final_eps=EXPLORATION_FINAL_EPS,

    policy_kwargs={"net_arch": [256, 256, 128]},
    verbose=1,
    tensorboard_log="runs/",
    device="auto",
)


# Train 
print(f"Training DQN for {TOTAL_TIMESTEPS:,} steps")
print(f"Replay buffer: {BUFFER_SIZE:,}  |  Batch size: {BATCH_SIZE}")
print(f"Exploration: {EXPLORATION_FRACTION*100:.0f}% of training, "
      f"final eps: {EXPLORATION_FINAL_EPS}")
print(f"Estimated time: ~30-45 minutes on CPU")

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[checkpoint_cb, eval_cb],
    tb_log_name="dqn",
    progress_bar=True,
)


# ── Save ──────────────────────────────────────────────────────────────────────
model.save("models/dqn_final")

print("\nDone.")
print("  Model saved to:       models_dqn/dqn_final.zip")
print("  Best checkpoint at:   models_dqn/dqn_best/best_model.zip")
print()
print("NEXT: evaluate with:")
print("  py -3 evaluate.py --model models_dqn/dqn_best/best_model "
      "--controllers random rule_based dqn --episodes 100")