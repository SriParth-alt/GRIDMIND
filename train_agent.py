"""
train_agent.py
Training loop: 1500-episode Dueling DQN with PER.

Data source is controlled by USE_REAL_DATA flag:
  False → synthetic data (generate_microgrid_data)
  True  → real ASHRAE building data (load_ashrae_data)

All hyperparameters live in config.py.
"""

import torch
import numpy as np
from collections import deque

from config import (
    STATE_BOUNDS, STATE_DIM, ACTION_DIM,
    EPISODES, N_DAYS_TRAIN, DATA_SEED,
    CHECKPOINT_FREQ, LOG_FREQ, REWARD_WINDOW,
    PLATEAU_THRESH, MODEL_PATH,
)
from data_generator import generate_microgrid_data
from ashrae_pipeline import load_ashrae_data
from environment.microgrid_env import MicrogridEnv
from agents.dqn_agent import DQNAgent

# ── Toggle this to switch between data sources ────────────────────────────── #
USE_REAL_DATA = True   # False = synthetic, True = ASHRAE real data
# ─────────────────────────────────────────────────────────────────────────── #


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


def load_data():
    if USE_REAL_DATA:
        return load_ashrae_data(n_days=N_DAYS_TRAIN, seed=DATA_SEED)
    return generate_microgrid_data(n_days=N_DAYS_TRAIN, seed=DATA_SEED)


def main():
    data  = load_data()
    env   = MicrogridEnv(data, shuffle_episodes=True)
    agent = DQNAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)

    reward_window = deque(maxlen=REWARD_WINDOW)
    cost_window   = deque(maxlen=REWARD_WINDOW)
    best_reward   = float("-inf")

    source = "ASHRAE real data" if USE_REAL_DATA else "synthetic data"
    print(f"Data source: {source}")
    print(f"State dim  : {STATE_DIM}D")
    print(f"Dataset    : {len(data)} rows ({data['day'].nunique()} days)\n")
    print(f"{'Ep':>5} | {'Reward':>10} | {'AvgR50':>10} | {'Cost':>8} | {'Eps':>7} | {'Loss':>8}")
    print("-" * 64)

    for ep in range(EPISODES):
        state        = state_to_vector(env.reset())
        total_reward = 0.0
        total_cost   = 0.0
        losses       = []
        done         = False

        while not done:
            action                         = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            next_vec                       = state_to_vector(next_state)

            agent.remember(state, action, reward, next_vec, done)
            loss = agent.train()
            if loss is not None:
                losses.append(loss)

            state         = next_vec
            total_reward += reward
            total_cost   += info["cost"]

        agent.decay_epsilon()

        reward_window.append(total_reward)
        cost_window.append(total_cost)
        avg_reward = np.mean(reward_window)
        avg_loss   = np.mean(losses) if losses else 0.0

        marker = ""
        if total_reward > best_reward:
            best_reward = total_reward
            marker = " *"

        if (ep + 1) % LOG_FREQ == 0:
            print(
                f"{ep+1:>5} | "
                f"{total_reward:>10.2f} | "
                f"{avg_reward:>10.2f} | "
                f"{total_cost:>8.2f} | "
                f"{agent.epsilon:>7.4f} | "
                f"{avg_loss:>8.4f}"
                f"{marker}"
            )

        if (ep + 1) % CHECKPOINT_FREQ == 0:
            ckpt = f"checkpoint_ep{ep+1}.pth"
            torch.save(agent.model.state_dict(), ckpt)
            print(f"  Checkpoint -> {ckpt}")

        if ep > 100 and (ep + 1) % CHECKPOINT_FREQ == 0:
            w = list(reward_window)
            if len(w) >= 50 and abs(np.mean(w[25:]) - np.mean(w[:25])) < PLATEAU_THRESH:
                print(f"  Warning: Plateau detected at ep {ep+1}")

    print(f"\nDone. Best reward: {best_reward:.2f}")
    torch.save(agent.model.state_dict(), MODEL_PATH)
    print(f"Model saved -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
