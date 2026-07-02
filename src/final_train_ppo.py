import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    BaseCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecFrameStack
from env_config import make_env

# ── Hyperparameters ───────────────────────────────────────────────────────────

TOTAL_TIMESTEPS = 500_000 
N_ENVS          = 4
N_STACK         = 4

def linear_schedule(initial_lr: float):
    def schedule(progress_remaining: float) -> float:
        return initial_lr * progress_remaining
    return schedule

LEARNING_RATE   = linear_schedule(3e-4)
N_STEPS         = 512
BATCH_SIZE      = 256
N_EPOCHS        = 10
GAMMA           = 0.99
GAE_LAMBDA      = 0.95
CLIP_RANGE      = 0.2
#ENT_COEF        = 0.05
ENT_COEF        = 0.01
EVAL_FREQ       = 20_000
N_EVAL_EPISODES = 30

os.makedirs("models/checkpoints", exist_ok=True)
os.makedirs("models/ppo_best",    exist_ok=True)
os.makedirs("runs",               exist_ok=True)


class SyncNormalizerCallback(BaseCallback):
    def __init__(self, train_env: VecNormalize, eval_env: VecNormalize, verbose=0):
        super().__init__(verbose)
        self.train_env = train_env
        self.eval_env  = eval_env

    def _on_step(self) -> bool:
        self.eval_env.obs_rms     = self.train_env.obs_rms
        self.eval_env.ret_rms     = self.train_env.ret_rms
        self.eval_env.clip_obs    = self.train_env.clip_obs
        self.eval_env.clip_reward = self.train_env.clip_reward
        return True


if __name__ == "__main__":
    
    # Use make_env directly to keep your exact env_config settings
    train_env = make_vec_env(make_env, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)
    train_env = VecFrameStack(train_env, n_stack=N_STACK)

    eval_env = make_vec_env(make_env, n_envs=1, vec_env_cls=DummyVecEnv)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    eval_env = VecFrameStack(eval_env, n_stack=N_STACK)

    sync_cb = SyncNormalizerCallback(train_env.venv, eval_env.venv)

    checkpoint_cb = CheckpointCallback(
        save_freq=100_000 // N_ENVS,
        save_path="models/checkpoints/",
        name_prefix="ppo_ckpt",
        verbose=1,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="models/ppo_best/",
        log_path="runs/ppo_eval/",
        eval_freq=EVAL_FREQ // N_ENVS,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        verbose=1,
    )

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
        #policy_kwargs={"net_arch": [256, 256, 128]},
        policy_kwargs={"net_arch": [512, 512, 256]},
        verbose=1,
        tensorboard_log="runs/",
        device="auto",
    )

    
    total_updates = TOTAL_TIMESTEPS // (N_STEPS * N_ENVS)
    print(f"Training for {TOTAL_TIMESTEPS:,} steps across {N_ENVS} parallel envs")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[sync_cb, checkpoint_cb, eval_cb],
        tb_log_name="ppo",
        progress_bar=True,
    )

    model.save("models/ppo_final")
    train_env.venv.save("models/vecnormalize.pkl")

    print("\nDone.")
    print("  Model saved to:       models/ppo_final.zip")
    print("  Normalizer saved to:  models/vecnormalize.pkl")
    print("  Best checkpoint at:   models/ppo_best/best_model.zip")

    train_env.close()
    eval_env.close()