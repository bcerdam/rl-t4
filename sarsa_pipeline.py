import numpy as np
from utils import e_greedy_policy


def sarsa(env, weights, feature_extractor, gamma, alpha, epsilon, possible_actions):
    observation, info = env.reset()
    action = e_greedy_policy(observation, weights, feature_extractor, epsilon, possible_actions)

    terminated = False
    truncated = False
    episode_length = 0
    while terminated is False and truncated is False:
        episode_length += 1
        next_observation, reward, terminated, truncated, info = env.step(action)
        next_action = e_greedy_policy(next_observation, weights, feature_extractor, epsilon, possible_actions)
        features = feature_extractor.get_features(observation, action)

        if terminated or truncated:
            delta = reward
        else:
            delta = reward + gamma * np.dot(weights, feature_extractor.get_features(next_observation, next_action))

        weights += alpha * (delta - np.dot(weights, features)) * features
        observation = next_observation
        action = next_action
    return weights, episode_length