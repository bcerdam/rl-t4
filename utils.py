import numpy as np
import matplotlib.pyplot as plt
import os


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