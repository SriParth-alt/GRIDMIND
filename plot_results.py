"""
plot_results.py
Visualise RL agent performance: cost, emissions, reward curves,
and an action distribution breakdown.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import STATE_BOUNDS, STATE_DIM, ACTION_DIM, MODEL_PATH
from ashrae_pipeline import load_ashrae_data
from environment.microgrid_env import MicrogridEnv
from agents.dqn_agent import DQNAgent


ACTION_LABELS = ["Idle", "Charge Solar", "Discharge", "Smart"]
COLORS        = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]


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


def plot_results(
    baseline_cost: float,
    rl_cost: float,
    model_path: str = MODEL_PATH,
    n_days: int = 30,
    baseline_emission: float = None,
    rl_emission: float = None,
):
    agent = DQNAgent(STATE_DIM, ACTION_DIM)
    agent.model.load_state_dict(torch.load(model_path, weights_only=True))
    agent.model.eval()
    agent.epsilon = 0.0

    episodes        = 30
    reward_history  = []
    cost_history    = []
    emission_history= []
    action_counts   = np.zeros(ACTION_DIM, dtype=int)

    for ep in range(episodes):
        data  = load_ashrae_data(n_days=n_days, seed=ep)
        env   = MicrogridEnv(data, shuffle_episodes=False)
        state = state_to_vector(env.reset())

        total_reward = 0.0
        total_cost   = 0.0
        done         = False

        while not done:
            action                        = agent.choose_action(state)
            next_state, reward, done, info= env.step(action)
            action_counts[action]        += 1
            total_reward                 += reward
            total_cost                   += info["cost"]
            state                         = state_to_vector(next_state)

        reward_history.append(total_reward)
        cost_history.append(total_cost)
        emission_history.append(env.total_emission)

    # ── Layout ─────────────────────────────────────────────────────────── #
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("AI Microgrid Optimization — Performance Dashboard", fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    ep_range = range(1, episodes + 1)

    # 1. Reward curve
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ep_range, reward_history, color=COLORS[0])
    ax1.axhline(np.mean(reward_history), linestyle="--", color="gray", linewidth=0.8, label=f"mean={np.mean(reward_history):.1f}")
    ax1.set_title("Reward per Episode")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.legend(fontsize=8)

    # 2. Cost curve
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(ep_range, cost_history, color=COLORS[2])
    ax2.axhline(np.mean(cost_history), linestyle="--", color="gray", linewidth=0.8, label=f"mean={np.mean(cost_history):.0f}")
    ax2.set_title("Grid Cost per Episode")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Cost ($)")
    ax2.legend(fontsize=8)

    # 3. Emission curve
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(ep_range, emission_history, color=COLORS[1])
    ax3.axhline(np.mean(emission_history), linestyle="--", color="gray", linewidth=0.8, label=f"mean={np.mean(emission_history):.0f}")
    ax3.set_title("CO₂ Emissions per Episode")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("kg CO₂")
    ax3.legend(fontsize=8)

    # 4. Cost comparison bar
    ax4 = fig.add_subplot(gs[1, 0])
    bars = ax4.bar(["Baseline", "RL Agent"], [baseline_cost, rl_cost], color=[COLORS[2], COLORS[0]])
    ax4.set_title("Cost Comparison")
    ax4.set_ylabel("Total Cost ($)")
    for bar, val in zip(bars, [baseline_cost, rl_cost]):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                 f"{val:.0f}", ha="center", va="bottom", fontsize=9)

    # 5. Emission comparison bar
    if baseline_emission is not None and rl_emission is not None:
        ax5 = fig.add_subplot(gs[1, 1])
        bars5 = ax5.bar(["Baseline", "RL Agent"], [baseline_emission, rl_emission], color=[COLORS[2], COLORS[1]])
        ax5.set_title("Emission Comparison")
        ax5.set_ylabel("kg CO₂")
        for bar, val in zip(bars5, [baseline_emission, rl_emission]):
            ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                     f"{val:.0f}", ha="center", va="bottom", fontsize=9)

    # 6. Action distribution pie
    ax6 = fig.add_subplot(gs[1, 2])
    wedges, texts, autotexts = ax6.pie(
        action_counts, labels=ACTION_LABELS, colors=COLORS,
        autopct="%1.1f%%", startangle=90, textprops={"fontsize": 8}
    )
    ax6.set_title("Action Distribution")

    plt.savefig("microgrid_results.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("Plot saved -> microgrid_results.png")
