import gymnasium as gym
import numpy as np
from tqdm import tqdm
from FeatureExtractor import FeatureExtractor
from actor_critic_pipeline import actor_critic
from utils import plot_single_result

env_name = "MountainCarContinuous-v0"
N_RUNS = 30
N_EPISODES = 1000
GAMMA = 1.0
CRITIC_LR = 0.001
ACTOR_LR = 0.0001

if __name__ == '__main__':
    print('Actor-Critic aprox. lineal...')

    all_episode_lengths_ac = np.zeros((N_RUNS, N_EPISODES))
    for run in tqdm(range(N_RUNS), desc="Total Runs"):
        env = gym.make(env_name)
        feature_extractor = FeatureExtractor()
        n_features = feature_extractor.num_of_features
        critic_lw = np.zeros(n_features)
        mean_lw = np.zeros(n_features)
        std_lw = np.zeros(n_features)

        for episode in range(N_EPISODES):
            critic_lw, mean_lw, std_lw, episode_length = actor_critic(
                env,
                feature_extractor,
                critic_lw,
                mean_lw,
                std_lw,
                GAMMA,
                CRITIC_LR,
                ACTOR_LR
            )
            all_episode_lengths_ac[run, episode] = episode_length
        env.close()

    plot_single_result(all_episode_lengths_ac, N_EPISODES)