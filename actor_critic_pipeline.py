import numpy as np
from utils import update_statistical_parameters


def actor_critic(env, feature_extractor, critic_lw, mean_lw, std_lw, gamma, critic_lr, actor_lr):
    observation, info = env.reset()
    observation_features = feature_extractor.get_features(observation)

    terminated = False
    truncated = False
    episode_length = 0
    while terminated is False and truncated is False:
        episode_length += 1
        mean = np.dot(mean_lw, observation_features)
        std = np.exp(np.dot(std_lw, observation_features))
        gaussian_action = np.random.normal(mean, std)
        gaussian_action_clipped = np.clip(gaussian_action, -1.0, 1.0)
        next_observation, reward, terminated, truncated, info = env.step([gaussian_action_clipped])
        next_observation_features = feature_extractor.get_features(next_observation)

        if terminated is False:
            error = reward + gamma * np.dot(critic_lw, next_observation_features) - np.dot(critic_lw, observation_features)
        else:
            error = reward + gamma * 0.0 - np.dot(critic_lw, observation_features)

        critic_lw = critic_lw + critic_lr * error * observation_features
        mean_lw, std_lw = update_statistical_parameters(mean_lw, std_lw, actor_lr, error, gaussian_action, mean, std, observation_features)
        observation_features = next_observation_features
    return critic_lw, mean_lw, std_lw, episode_length