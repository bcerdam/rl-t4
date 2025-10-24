import os
import shutil
import gymnasium as gym
import numpy as np
from tqdm import tqdm
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from utils import plot_sb3_results

env_name = "MountainCar-v0"
N_RUNS = 1
N_TIMESTEPS = 300000
dqn_results_dir = "dqn_results"

if __name__ == '__main__':
    if os.path.exists(dqn_results_dir):
        shutil.rmtree(dqn_results_dir)
    os.makedirs(dqn_results_dir, exist_ok=True)

    hyperparameters = {
        "policy": 'MlpPolicy',
        "learning_rate": 0.01,
        "batch_size": 256,
        "buffer_size": 1000000,
        "learning_starts": 2000,
        "gamma": 0.95,
        "target_update_interval": 50,
        "train_freq": 32,
        "gradient_steps": 16,
        "exploration_fraction": 0.1,
        "exploration_final_eps": 0.05,
        "policy_kwargs": {"net_arch": [128, 128]}
    }

    for run in tqdm(range(N_RUNS), desc="runs"):
        run_log_path = os.path.join(dqn_results_dir, f"run_{run}")
        os.makedirs(run_log_path, exist_ok=True)
        env = gym.make(env_name)
        env = Monitor(env, filename=run_log_path)
        model = DQN(env=env, **hyperparameters)
        model.learn(total_timesteps=N_TIMESTEPS)
        env.close()

    plot_sb3_results(dqn_results_dir, N_RUNS, 1500, "DQN", env_name, "resultado_pregunta_b")
    shutil.rmtree(dqn_results_dir)