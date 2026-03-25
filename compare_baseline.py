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
# RL run
# -------------------------------
def run_rl(agent):
    data = generate_microgrid_data(n_days=2, seed=42)
    env = MicrogridEnv(data, shuffle_episodes=False)

    state = state_to_vector(env.reset())
    total_cost = 0

    done = False
    while not done:
        action = agent.choose_action(state)
        next_state, _, done, info = env.step(action)

        total_cost += info["cost"]
        state = state_to_vector(next_state)

    return total_cost


# -------------------------------
# BASELINE (simple)
# -------------------------------
def run_baseline():
    data = generate_microgrid_data(n_days=2, seed=42)
    env = MicrogridEnv(data, shuffle_episodes=False)

    state = env.reset()
    total_cost = 0

    done = False
    while not done:
        action = 0  # always grid
        next_state, _, done, info = env.step(action)

        total_cost += info["cost"]
        state = next_state

    return total_cost


# -------------------------------
# MAIN FUNCTION (IMPORTANT)
# -------------------------------
def compare_with_baseline(model_path):

    # Load trained agent
    agent = DQNAgent(6, 4)
    agent.model.load_state_dict(torch.load(model_path))
    agent.model.eval()
    agent.epsilon = 0

    # Run both
    rl_cost = run_rl(agent)
    baseline_cost = run_baseline()

    print("\n===== COMPARISON =====")
    print(f"Without RL (Baseline): {baseline_cost:.2f}")
    print(f"With RL (Agent):      {rl_cost:.2f}")
    print("======================")

    improvement = ((baseline_cost - rl_cost) / baseline_cost) * 100
    print(f"\n💡 Cost Reduction: {improvement:.2f}%")

    # 🔥 RETURN VALUES (CRITICAL)
    return baseline_cost, rl_cost


# -------------------------------
# STANDALONE RUN (OPTIONAL)
# -------------------------------
if __name__ == "__main__":
    compare_with_baseline("microgrid_dqn.pth")