"""
main.py
Entry point: evaluate trained agent, generate plots, run demo.
Run train_agent.py first to produce microgrid_dqn.pth.
"""

from config import MODEL_PATH
from compare_baseline import compare_with_baseline
from demo import run_demo
from plot_results import plot_results


def main():
    print("\n========================================")
    print("  AI MICROGRID OPTIMIZATION SYSTEM")
    print("========================================\n")

    n_days = int(input("Enter number of days for simulation: "))

    print("\n[1/3] Comparing RL agent vs baseline...\n")
    baseline_cost, rl_cost, baseline_emission, rl_emission = compare_with_baseline(
        model_path=MODEL_PATH, n_days=n_days
    )

    print("\n[2/3] Generating performance plots...\n")
    plot_results(
        baseline_cost, rl_cost,
        model_path=MODEL_PATH,
        n_days=n_days,
        baseline_emission=baseline_emission,
        rl_emission=rl_emission,
    )

    print("\n[3/3] Running 2-day demo...\n")
    run_demo(MODEL_PATH)

    print("\nPipeline complete.\n")


if __name__ == "__main__":
    main()
