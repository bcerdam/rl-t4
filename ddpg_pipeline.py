import os
import shutil
import gymnasium as gym
import numpy as np
from tqdm import tqdm
from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from utils import plot_sb3_results

env_name = "MountainCarContinuous-v0"
N_RUNS = 10
N_TIMESTEPS = 300000
ddpg_results_dir = "ddpg_results"


# Se demoro 22 min en 1 run.
if __name__ == '__main__':
    if os.path.exists(ddpg_results_dir):
        shutil.rmtree(ddpg_results_dir)
    os.makedirs(ddpg_results_dir, exist_ok=True)

    env_for_noise = gym.make(env_name)
    number_of_actions = env_for_noise.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(number_of_actions), sigma=0.75 * np.ones(number_of_actions))
    env_for_noise.close()
    hyperparameters = {"policy": 'MlpPolicy', "action_noise": action_noise}
    for run in tqdm(range(N_RUNS), desc="runs"):
        run_log_path = os.path.join(ddpg_results_dir, f"run_{run}")
        os.makedirs(run_log_path, exist_ok=True)
        env = gym.make(env_name)
        env = Monitor(env, filename=run_log_path)
        model = DDPG(env=env, **hyperparameters)
        model.learn(total_timesteps=N_TIMESTEPS)
        env.close()

    plot_sb3_results(ddpg_results_dir, N_RUNS, 1000, "DDPG", env_name, "resultado_pregunta_d")
    shutil.rmtree(ddpg_results_dir)