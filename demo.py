import torch
import numpy as np

from environment.microgrid_env import MicrogridEnv
from agents.dqn_agent import DQNAgent
from data_generator import generate_microgrid_data


# -------------------------------
# State normalization
# -------------------------------
def state_to_vector(state):
    return np.array([
        state["demand"] / 200,
        state["solar"] / 160,
        state["price"] / 25,
        state["next_price"] / 25,
        state["battery_soc"] / 100,
        state["hour"] / 24
    ], dtype=np.float32)


# -------------------------------
# MAIN DEMO FUNCTION
# -------------------------------
def run_demo(model_path):

    print("\n🔌 LOADING TRAINED MODEL...\n")

    # Load small dataset for demo
    data = generate_microgrid_data(n_days=2, seed=42)
    env = MicrogridEnv(data, shuffle_episodes=False)

    # Load agent
    agent = DQNAgent(state_dim=6, action_dim=4)
    agent.model.load_state_dict(torch.load(model_path))
    agent.model.eval()

    # Disable randomness for demo
    agent.epsilon = 0

    state = state_to_vector(env.reset())

    print("===== MICROGRID AI DEMO =====\n")
    print("Hr | Demand | Solar | Price | Action | Grid | Battery | SOC")
    print("-" * 70)

    total_cost = 0

    done = False
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, info = env.step(action)

        total_cost += info["cost"]

        print(
            f"{int(next_state['hour']):>2} | "
            f"{next_state['demand']:>6.1f} | "
            f"{next_state['solar']:>5.1f} | "
            f"{next_state['price']:>5.2f} | "
            f"{action:^6} | "
            f"{info['grid_used']:>5} | "
            f"{info['battery_used']:>7} | "
            f"{info['battery_soc']:>6}"
        )

        state = state_to_vector(next_state)

    print("\n==============================")
    print(f"💰 Total Cost (AI): {total_cost:.2f}")
    print("==============================\n")


# -------------------------------
# STANDALONE RUN (OPTIONAL)
# -------------------------------
if __name__ == "__main__":
    run_demo("microgrid_dqn.pth")