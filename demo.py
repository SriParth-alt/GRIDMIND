"""
demo.py
2-day walkthrough showing the trained agent's hour-by-hour decisions.
"""

import torch
import numpy as np

from config import STATE_BOUNDS, STATE_DIM, ACTION_DIM, MODEL_PATH
from environment.microgrid_env import MicrogridEnv
from agents.dqn_agent import DQNAgent
from ashrae_pipeline import load_ashrae_data


ACTION_NAMES = {0: "Idle", 1: "ChargeSolar", 2: "Discharge", 3: "Smart"}


def state_to_vector(state: dict) -> np.ndarray:
    return np.array([
        state["demand"]            / STATE_BOUNDS["demand"],
        state["solar"]             / STATE_BOUNDS["solar"],
        state["price"]             / STATE_BOUNDS["price"],
        state["next_price"]        / STATE_BOUNDS["next_price"],
        state["next_demand"]       / STATE_BOUNDS["next_demand"],
        state["next_solar"]        / STATE_BOUNDS["next_solar"],
        state["battery_soc"]       / STATE_BOUNDS["battery_soc"],
        state["hour"]              / STATE_BOUNDS["hour"],
        state["day_of_week"]       / STATE_BOUNDS["day_of_week"],
        np.clip(state["demand_delta"] / STATE_BOUNDS["demand_delta"], -1.0, 1.0),
        state["next24_peak_demand"] / STATE_BOUNDS["next24_peak_demand"],
        state["next24_peak_hour"]   / STATE_BOUNDS["next24_peak_hour"],
        state["next24_avg_demand"]  / STATE_BOUNDS["next24_avg_demand"],
    ], dtype=np.float32)


def run_demo(model_path: str = MODEL_PATH):
    data  = load_ashrae_data(n_days=2, seed=42)
    env   = MicrogridEnv(data, shuffle_episodes=False)

    agent = DQNAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)
    agent.model.load_state_dict(torch.load(model_path, weights_only=True))
    agent.model.eval()
    agent.epsilon = 0.0

    state      = state_to_vector(env.reset())
    total_cost = 0.0
    total_emit = 0.0

    print("\n===== MICROGRID AI DEMO (2 days) =====\n")
    header = f"{'Hr':>2} | {'Demand':>6} | {'Solar':>5} | {'Price':>5} | {'Action':>12} | {'Grid':>5} | {'Batt':>5} | {'SOC':>5} | {'CO2':>5}"
    print(header)
    print("-" * len(header))

    done = False
    while not done:
        action                        = agent.choose_action(state)
        next_state, reward, done, info= env.step(action)

        total_cost += info["cost"]
        total_emit += info["emission"]

        print(
            f"{int(next_state['hour']):>2} | "
            f"{next_state['demand']:>6.1f} | "
            f"{next_state['solar']:>5.1f} | "
            f"{next_state['price']:>5.2f} | "
            f"{ACTION_NAMES[action]:>12} | "
            f"{info['grid_used']:>5.1f} | "
            f"{info['battery_used']:>5.1f} | "
            f"{info['battery_soc']:>5.1f} | "
            f"{info['emission']:>5.2f}"
        )

        state = state_to_vector(next_state)

    print(f"\nTotal cost      : {total_cost:.2f} $")
    print(f"Total emissions : {total_emit:.2f} kg CO2")


if __name__ == "__main__":
    run_demo()
