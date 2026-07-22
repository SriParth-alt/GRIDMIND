"""
compare_baseline.py
Compare trained RL agent against a rule-based heuristic baseline.
Returns (baseline_cost, rl_cost, baseline_emission, rl_emission).
"""

import numpy as np
import torch

from config import STATE_BOUNDS, STATE_DIM, ACTION_DIM, MODEL_PATH
from ashrae_pipeline import load_ashrae_data
from environment.microgrid_env import MicrogridEnv
from agents.dqn_agent import DQNAgent


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


def _baseline_action(state: dict, price_low: float = 2.05, price_high: float = 3.49) -> int:
    """
    Rule-based heuristic using price arbitrage. Thresholds default to the
    25th/75th percentiles of real 2016 NYISO prices; callers with different
    data should pass their own percentiles.
      - Price cheap (bottom quartile) + battery has room → charge (action 1)
      - Price expensive (top quartile) + battery charged → discharge (action 2)
      - Otherwise                                        → idle (action 0)
    """
    net_demand = max(0.0, state["demand"] - state["solar"])

    if state["price"] <= price_low and state["battery_soc"] < 85:
        return 1   # charge from cheap grid

    if net_demand > 0 and state["price"] >= price_high and state["battery_soc"] > 20:
        return 2   # discharge when grid is expensive

    return 0       # idle — let grid cover demand


def run_rl(agent: DQNAgent, n_days: int) -> tuple:
    data = _load_data(n_days)
    env  = MicrogridEnv(data, shuffle_episodes=False)

    state      = state_to_vector(env.reset())
    total_cost = 0.0
    done       = False

    while not done:
        action                    = agent.choose_action(state)
        next_state, _, done, info = env.step(action)
        total_cost               += info["cost"]
        state                     = state_to_vector(next_state)

    return total_cost, env.total_emission


def run_baseline(n_days: int) -> tuple:
    data = _load_data(n_days)
    env  = MicrogridEnv(data, shuffle_episodes=False)

    state      = env.reset()
    total_cost = 0.0
    done       = False

    while not done:
        action = _baseline_action(state, env.price_low, env.price_high)
        next_state, _, done, info = env.step(action)
        total_cost               += info["cost"]
        state                     = next_state

    return total_cost, env.total_emission


DATA_SEED = 42


def _load_data(n_days: int):
    return load_ashrae_data(n_days=n_days, seed=DATA_SEED)


def compare_with_baseline(model_path: str = MODEL_PATH, n_days: int = 30):
    agent = DQNAgent(STATE_DIM, ACTION_DIM)
    agent.model.load_state_dict(torch.load(model_path, weights_only=True))
    agent.model.eval()
    agent.epsilon = 0.0

    rl_cost,       rl_emission       = run_rl(agent, n_days)
    baseline_cost, baseline_emission = run_baseline(n_days)

    cost_imp     = (baseline_cost     - rl_cost)     / baseline_cost     * 100
    emission_imp = (baseline_emission - rl_emission) / baseline_emission * 100

    print("\n===== COMPARISON =====")
    print(f"Baseline cost     : {baseline_cost:.2f}")
    print(f"RL agent cost     : {rl_cost:.2f}  ({cost_imp:+.1f}%)")
    print(f"Baseline emission : {baseline_emission:.2f} kg CO2")
    print(f"RL agent emission : {rl_emission:.2f} kg CO2  ({emission_imp:+.1f}%)")
    print("======================")

    return baseline_cost, rl_cost, baseline_emission, rl_emission


if __name__ == "__main__":
    n_days = int(input("Enter number of days: "))
    compare_with_baseline(n_days=n_days)
