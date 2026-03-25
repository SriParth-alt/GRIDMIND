import matplotlib.pyplot as plt
import numpy as np
import torch

from data_generator import generate_microgrid_data
from environment.microgrid_env import MicrogridEnv
from agents.dqn_agent import DQNAgent


# -------------------------------
# State normalization
# -------------------------------
def state_to_vector(state):
    return np.array([
        state["demand"]/200,
        state["solar"]/160,
        state["price"]/25,
        state["next_price"]/25,
        state["battery_soc"]/100,
        state["hour"]/24
    ], dtype=np.float32)


# -------------------------------
# MAIN FUNCTION
# -------------------------------
def plot_results(baseline_cost, rl_cost, model_path):

    # 🔹 1. BAR CHART (Cost Comparison)
    plt.figure()
    labels = ['Baseline', 'RL Agent']
    costs = [baseline_cost, rl_cost]

    plt.bar(labels, costs)
    plt.title("Cost Comparison")
    plt.ylabel("Total Cost")


    # 🔹 2. LOAD MODEL FOR EPISODE EVALUATION
    agent = DQNAgent(6, 4)
    agent.model.load_state_dict(torch.load(model_path))
    agent.model.eval()
    agent.epsilon = 0

    # 🔹 3. RUN MULTIPLE EPISODES
    episodes = 30
    reward_history = []
    cost_history = []

    for ep in range(episodes):
        data = generate_microgrid_data(n_days=2, seed=ep)
        env = MicrogridEnv(data, shuffle_episodes=False)

        state = state_to_vector(env.reset())
        total_reward = 0
        total_cost = 0

        done = False
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)

            total_reward += reward
            total_cost += info["cost"]

            state = state_to_vector(next_state)

        reward_history.append(total_reward)
        cost_history.append(total_cost)

    # 🔹 4. REWARD GRAPH
    plt.figure()
    plt.plot(reward_history)
    plt.title("Reward over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Reward")

    # 🔹 5. COST GRAPH
    plt.figure()
    plt.plot(cost_history)
    plt.title("Cost over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Cost")

    plt.show()