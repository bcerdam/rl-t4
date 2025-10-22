import numpy as np
import matplotlib.pyplot as plt
import os
import csv


def update_statistical_parameters(mean_lw, std_lw, actor_lr, error, gaussian_action, mean, std, observation_features):
    mean_lw = mean_lw + actor_lr * error * (gaussian_action - mean) / (std ** 2) * observation_features
    std_lw = std_lw + actor_lr * error * (((gaussian_action - mean) ** 2) / (std ** 2) - 1) * observation_features
    return mean_lw, std_lw


def plot_single_result(all_runs_lengths, n_episodes):
    mean_lengths_over_runs = np.mean(all_runs_lengths, axis=0)
    grouped_means = np.mean(mean_lengths_over_runs.reshape(-1, 10), axis=1)
    x_axis = np.arange(10, n_episodes + 1, 10)
    plt.figure(figsize=(12, 8))
    plt.plot(x_axis, grouped_means, label="Actor-Critic con aprox. lineal")
    plt.xlabel("Episodio")
    plt.ylabel(f"Largo Promedio de Episodio ")
    plt.title("Actor-Critic con aproximacion lineal: MountainCarContinuous-v0")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figuras/c_actor_critic.jpeg", dpi=500)


def e_greedy_policy(observation, weights, feature_extractor, epsilon, possible_actions):
    random_decimal = np.random.uniform(0, 1)
    if random_decimal < epsilon:
        return np.random.choice(possible_actions)
    else:
        q_values = []
        for action in possible_actions:
            q_values.append(np.dot(weights, feature_extractor.get_features(observation, action)))
        return np.argmax(q_values)


def plot_comparison_results(sarsa_lengths, q_learning_lengths, n_episodes):
    avg_sarsa = np.mean(sarsa_lengths, axis=0)
    avg_q_learning = np.mean(q_learning_lengths, axis=0)
    plt.figure(figsize=(12, 8))
    episodes = np.arange(1, n_episodes + 1)
    plt.plot(episodes, avg_sarsa, label='Largo Promedio de Episodio (SARSA)')
    plt.plot(episodes, avg_q_learning, label='Largo Promedio de Episodio (Q-Learning)', color='orange')
    plt.title('SARSA vs. Q-Learning con Aproximacion Lineal')
    plt.xlabel('Episodio')
    plt.ylabel('Largo Promedio de Episodio')
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=100)
    plt.tight_layout()
    plt.savefig("figuras/pregunta_a_sarsa_vs_qlearning.jpeg", dpi=500)
    plt.show()