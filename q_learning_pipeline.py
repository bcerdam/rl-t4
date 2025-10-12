import numpy as np
from utils import e_greedy_policy


def q_learning(env, weights, feature_extractor, gamma, alpha, epsilon, possible_actions):
    observation, info = env.reset()

    terminated = False
    truncated = False
    episode_length = 0
    while terminated is False and truncated is False:
        episode_length += 1
        action = e_greedy_policy(observation, weights, feature_extractor, epsilon, possible_actions)
        next_observation, reward, terminated, truncated, info = env.step(action)
        features = feature_extractor.get_features(observation, action)

        if terminated or truncated:
            delta = reward
        else:
            next_state_q_values = []
            for action in possible_actions:
                next_state_q_values.append(np.dot(weights, feature_extractor.get_features(next_observation, action)))
            delta = reward + gamma * np.max(next_state_q_values)
        weights += alpha * (delta - np.dot(weights, features)) * features
        observation = next_observation
    return weights, episode_length