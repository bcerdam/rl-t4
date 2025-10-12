import gymnasium as gym
import numpy as np
from tqdm import tqdm
from FeatureExtractor import FeatureExtractor
from sarsa_pipeline import sarsa
from q_learning_pipeline import q_learning
from utils import plot_comparison_results

env_name = "MountainCar-v0"
N_RUNS = 30
N_EPISODES = 1000
GAMMA = 1.0
ALPHA = 0.5/8
EPSILON = 0.0
POSSIBLE_ACTIONS = [0, 1, 2]

if __name__ == '__main__':
    print('SARSA aprox. lineal...')

    all_episode_lengths_sarsa_la = np.zeros((N_RUNS, N_EPISODES))
    for run in tqdm(range(N_RUNS), desc="Total Runs"):
        env = gym.make(env_name)
        feature_extractor = FeatureExtractor()
        n_features = feature_extractor.num_of_features
        weights = np.zeros(n_features)

        for episode in range(N_EPISODES):
            weights, episode_length = sarsa(
                env,
                weights,
                feature_extractor,
                GAMMA,
                ALPHA,
                EPSILON,
                POSSIBLE_ACTIONS
            )
            all_episode_lengths_sarsa_la[run, episode] = episode_length
        env.close()

    print('Q-Learning aprox. lineal...')
    all_episode_lengths_q_learning_la = np.zeros((N_RUNS, N_EPISODES))
    for run in tqdm(range(N_RUNS), desc="Q-Learning Runs"):
        env = gym.make(env_name)
        feature_extractor = FeatureExtractor()
        n_features = feature_extractor.num_of_features
        weights = np.zeros(n_features)

        for episode in range(N_EPISODES):
            weights, episode_length = q_learning(
                env, weights, feature_extractor, GAMMA, ALPHA, EPSILON, POSSIBLE_ACTIONS
            )
            all_episode_lengths_q_learning_la[run, episode] = episode_length
        env.close()

    plot_comparison_results(
        all_episode_lengths_sarsa_la,
        all_episode_lengths_q_learning_la,
        N_EPISODES
    )