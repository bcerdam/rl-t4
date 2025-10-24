import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv


def plot_sb3_results(log_dir, num_runs, n_episodes, algorithm_name, env_name, plot_filename_base):
    all_runs_lengths_array = np.zeros((num_runs, n_episodes))
    all_runs_rewards_array = np.zeros((num_runs, n_episodes))

    for i in range(num_runs):
        file_path = os.path.join(log_dir, f"run_{i}", "monitor.csv")
        df = pd.read_csv(file_path, skiprows=1)

        run_lengths = df['l'].to_numpy()
        run_rewards = df['r'].to_numpy()

        num_eps_in_run = len(run_lengths)

        if num_eps_in_run >= n_episodes:
            all_runs_lengths_array[i, :] = run_lengths[:n_episodes]
            all_runs_rewards_array[i, :] = run_rewards[:n_episodes]
        else:
            all_runs_lengths_array[i, :num_eps_in_run] = run_lengths
            all_runs_lengths_array[i, num_eps_in_run:] = run_lengths[-1]

            all_runs_rewards_array[i, :num_eps_in_run] = run_rewards
            all_runs_rewards_array[i, num_eps_in_run:] = run_rewards[-1]

    mean_lengths_over_runs = np.mean(all_runs_lengths_array, axis=0)
    grouped_mean_lengths = np.mean(mean_lengths_over_runs.reshape(-1, 10), axis=1)

    mean_rewards_over_runs = np.mean(all_runs_rewards_array, axis=0)
    grouped_mean_rewards = np.mean(mean_rewards_over_runs.reshape(-1, 10), axis=1)

    x_axis = np.arange(10, n_episodes + 1, 10)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    ax1.plot(x_axis, grouped_mean_lengths, label=f"{algorithm_name} (Stable Baselines)")
    ax1.set_ylabel(f"Largo Promedio de Episodio")
    ax1.set_title(f" {algorithm_name} en {env_name}")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(x_axis, grouped_mean_rewards, label=f"{algorithm_name} (Stable Baselines)", color='orange')
    ax2.set_xlabel("Episodios")
    ax2.set_ylabel(f"Recompensa Promedio de Episodio")
    ax2.legend()
    ax2.grid(True)

    os.makedirs("figuras", exist_ok=True)
    plot_filename = f"figuras/{plot_filename_base}.jpeg"

    plt.tight_layout()
    plt.savefig(plot_filename, dpi=500, format='jpeg')
    print(f"Plot saved to {plot_filename}")

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