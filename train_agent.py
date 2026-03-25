"""
train_agent.py
Training loop with:
  - Realistic dataset from data_generator (60 days x 24 hours)
  - 6-dimensional normalised state vector (includes next_price)
  - Rolling reward window to track genuine improvement
  - Plateau detection warning every 100 episodes
  - Model saved to microgrid_dqn.pth after training completes
"""

import torch
import numpy as np
import pandas as pd
from collections import deque

from data_generator import generate_microgrid_data
from environment.microgrid_env import MicrogridEnv
from agents.dqn_agent import DQNAgent


# ──────────────────────────────────────────────────────────────────────── #
#  State normalisation                                                     #
# ──────────────────────────────────────────────────────────────────────── #
STATE_BOUNDS = {
    "demand":      200.0,
    "solar":       160.0,
    "price":        25.0,
    "next_price":   25.0,
    "battery_soc": 100.0,
    "hour":         24.0,
}

def state_to_vector(state: dict) -> np.ndarray:
    return np.array([
        state["demand"]      / STATE_BOUNDS["demand"],
        state["solar"]       / STATE_BOUNDS["solar"],
        state["price"]       / STATE_BOUNDS["price"],
        state["next_price"]  / STATE_BOUNDS["next_price"],
        state["battery_soc"] / STATE_BOUNDS["battery_soc"],
        state["hour"]        / STATE_BOUNDS["hour"],
    ], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────── #
#  TRAIN FUNCTION (UPDATED)                                                #
# ──────────────────────────────────────────────────────────────────────── #
def train_agent():

    data  = generate_microgrid_data(n_days=60, seed=42)
    env   = MicrogridEnv(data, shuffle_episodes=True)
    agent = DQNAgent(state_dim=6, action_dim=4)

    episodes      = 1500
    reward_window = deque(maxlen=50)
    cost_window   = deque(maxlen=50)
    best_reward   = float("-inf")

    projected = agent.epsilon * (agent.epsilon_decay ** episodes)
    print(f"Epsilon: 1.0 -> {projected:.4f} over {episodes} episodes")
    print(f"Dataset: {len(data)} rows ({data['day'].nunique()} days)\n")
    print(f"{'Ep':>5} | {'Reward':>10} | {'AvgR50':>10} | {'Cost':>8} | {'e':>7} | {'Loss':>8}")
    print("-" * 62)

    for ep in range(episodes):
        state        = state_to_vector(env.reset())
        total_reward = 0.0
        total_cost   = 0.0
        losses       = []
        done         = False

        while not done:
            action                         = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            next_state                     = state_to_vector(next_state)

            agent.remember(state, action, reward, next_state, done)
            loss = agent.train()
            if loss is not None:
                losses.append(loss)

            state         = next_state
            total_reward += reward
            total_cost   += info.get("cost", 0.0)

        agent.decay_epsilon()

        reward_window.append(total_reward)
        cost_window.append(total_cost)
        avg_reward = np.mean(reward_window)
        avg_loss   = np.mean(losses) if losses else 0.0

        marker = ""
        if total_reward > best_reward:
            best_reward = total_reward
            marker = " *"

        if (ep + 1) % 5 == 0:
            print(
                f"{ep+1:>5} | "
                f"{total_reward:>10.2f} | "
                f"{avg_reward:>10.2f} | "
                f"{total_cost:>8.2f} | "
                f"{agent.epsilon:>7.4f} | "
                f"{avg_loss:>8.4f}"
                f"{marker}"
            )

        if (ep + 1) % 100 == 0:
            torch.save(agent.model.state_dict(), f"checkpoint_ep{ep+1}.pth")
            print(f"  Checkpoint saved -> checkpoint_ep{ep+1}.pth")

        if ep > 100 and (ep + 1) % 100 == 0:
            w = list(reward_window)
            if abs(np.mean(w[25:]) - np.mean(w[:25])) < 0.5:
                print(f"  Warning: Plateau detected at ep {ep+1}")

    print(f"\nDone. Best reward: {best_reward:.2f}")

    # 🔥 SAVE MODEL
    model_path = "microgrid_dqn.pth"
    torch.save(agent.model.state_dict(), model_path)

    print(f"Model saved -> {model_path}")

    # 🔥 RETURN FOR MAIN PIPELINE
    return model_path


# ──────────────────────────────────────────────────────────────────────── #
#  RUN STANDALONE                                                          #
# ──────────────────────────────────────────────────────────────────────── #
if __name__ == "__main__":
    train_agent()