"""
Training script for QWOP using StableBaselines3 DQN algorithm.

This script demonstrates how to train a DQN agent on the QWOP environment
using hyperparameters adapted from the original qwop-gym project.

Recommended training duration: 20M+ timesteps for competitive results.
The default configuration uses 10M timesteps as a reasonable starting point.

Expected training time (approximate):
  - 10M timesteps: ~2-3 hours on a modern CPU
  - 20M timesteps: ~4-6 hours on a modern CPU
"""

import os

import qwop_gym_leaner  # noqa: F401 - Required to register QWOP-v1 environment
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecVideoRecorder


# Environment configuration
env_str = "QWOP-v1"
log_dir = "./logs/{}".format(env_str)

# Environment parameters (from original qwop-gym)
env_kwargs_dict = {
    # Paths to browser and chromedriver - will use defaults from PATH if None
    "browser": None,  # Will use 'chrome_for_testing' from PATH
    "driver": None,  # Will use 'chromedriver_for_testing' from PATH
    # Reward shaping parameters
    "failure_cost": 10,
    "success_reward": 50,
    "time_cost_mult": 10,
    # Performance parameters
    "frames_per_step": 4,  # Frameskip for faster training
    # Visual feedback during training
    "stat_in_browser": False,
    "game_in_browser": False,  # Disable rendering for performance
    "text_in_browser": r"Training in progress... Do not close this window.",
    # Action space
    "reduced_action_set": False,  # Use all 16 actions
    # Logging
    "loglevel": "WARN",
}

# Maximum steps per episode
max_episode_steps = 5000

# Create training environment with TimeLimit wrapper
env = make_vec_env(
    env_str,
    n_envs=1,
    env_kwargs=env_kwargs_dict,
    wrapper_class=TimeLimit,
    wrapper_kwargs={"max_episode_steps": max_episode_steps},
)

# Create evaluation environment
env_val = make_vec_env(
    env_str,
    n_envs=1,
    env_kwargs=env_kwargs_dict,
    wrapper_class=TimeLimit,
    wrapper_kwargs={"max_episode_steps": max_episode_steps},
)

# Callbacks for monitoring and checkpointing
# 1. Evaluation callback - monitors progress and saves best model
eval_callback = EvalCallback(
    env_val,
    best_model_save_path=log_dir,
    log_path=log_dir,
    eval_freq=50_000,  # Evaluate every 50k steps
    render=False,
    n_eval_episodes=10,
    deterministic=True,
    verbose=1,
)

# 2. Checkpoint callback - saves model at regular intervals
checkpoint_callback = CheckpointCallback(
    save_freq=500_000,  # Save every 500k steps
    save_path=os.path.join(log_dir, "checkpoints"),
    name_prefix="dqn_qwop",
    save_replay_buffer=True,  # Save replay buffer for resuming training
    verbose=1,
)

# DQN hyperparameters (adapted from original qwop-gym)
model = DQN(
    "MlpPolicy",  # Use MLP policy for numeric observations
    env,
    verbose=1,
    # Core DQN parameters
    buffer_size=100_000,
    learning_rate=3e-3,
    learning_starts=100_000,  # Start learning after 100k steps
    batch_size=64,
    tau=1.0,  # Hard update
    gamma=0.995,  # Discount factor
    # Training parameters
    train_freq=4,
    gradient_steps=1,
    target_update_interval=100_000,
    # Exploration parameters
    exploration_fraction=0.01,  # Explore for 1% of total timesteps
    exploration_initial_eps=1.0,
    exploration_final_eps=0.01,
    tensorboard_log=log_dir,
)

print("=" * 80)
print("QWOP DQN Training Configuration")
print("=" * 80)
print(f"Environment: {env_str}")
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")
print(f"Logs directory: {log_dir}")
print(f"Max episode steps: {max_episode_steps}")
print("=" * 80)

# Train the model
# Original qwop-gym recommends 20M+ timesteps for competitive results
# This configuration uses 10M as a reasonable starting point
# Adjust total_timesteps based on your needs:
#   - Quick test: 1M timesteps (~10-15 minutes)
#   - Reasonable agent: 10M timesteps (~2-3 hours)
#   - Competitive agent: 20M+ timesteps (~4-6+ hours)
total_timesteps = 10_000_000

print(f"Training for {total_timesteps:,} timesteps...")
print(f"Estimated time: ~{total_timesteps / 5_000_000:.1f}-{total_timesteps / 3_333_333:.1f} hours")
print("=" * 80)

model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=[eval_callback, checkpoint_callback])

print("=" * 80)
print("Training complete!")
print("=" * 80)

# Save the final model
model.save(os.path.join(log_dir, "dqn_qwop_final"))
print(f"✓ Final model saved to {log_dir}/dqn_qwop_final.zip")

# Evaluate the final model
print("\nEvaluating final model...")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20, deterministic=True)
print(f"Final model - Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Close training environments
env.close()
env_val.close()

# Load and evaluate the best model
print("\n" + "=" * 80)
print("Loading best model for final evaluation...")
print("=" * 80)
best_model_path = os.path.join(log_dir, "best_model")
env_test = make_vec_env(
    env_str,
    n_envs=1,
    seed=0,
    env_kwargs=env_kwargs_dict,
    wrapper_class=TimeLimit,
    wrapper_kwargs={"max_episode_steps": max_episode_steps},
)

best_model = DQN.load(best_model_path, env=env_test)

mean_reward, std_reward = evaluate_policy(best_model, env_test, n_eval_episodes=20, deterministic=True)
print(f"Best model - Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Record a video of the best model
print("\nRecording video of best model...")
video_dir = "./videos/"
os.makedirs(video_dir, exist_ok=True)

env_test = VecVideoRecorder(
    env_test,
    video_dir,
    video_length=max_episode_steps,
    record_video_trigger=lambda x: x == 0,
    name_prefix="best_model_qwop_dqn",
)

obs = env_test.reset()
for _ in range(max_episode_steps):
    action, _states = best_model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env_test.step(action)
    if dones:
        break

env_test.close()
print(f"✓ Video saved to {video_dir}")

print("\n" + "=" * 80)
print("All done! Summary:")
print("=" * 80)
print(f"✓ Final model: {log_dir}/dqn_qwop_final.zip")
print(f"✓ Best model: {log_dir}/best_model.zip")
print(f"✓ Checkpoints: {log_dir}/checkpoints/")
print(f"✓ TensorBoard logs: {log_dir}/")
print(f"✓ Video: {video_dir}")
print("\nTo view training progress:")
print(f"  tensorboard --logdir={log_dir}")
print("=" * 80)
