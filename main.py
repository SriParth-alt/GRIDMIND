from compare_baseline import compare_with_baseline
from demo import run_demo
from plot_results import plot_results


def main():

    # ✅ DEFINE MODEL PATH (THIS WAS MISSING)
    model_path = "microgrid_dqn.pth"

    print("\n========================================")
    print("⚡ AI MICROGRID OPTIMIZATION SYSTEM")
    print("========================================\n")

    print("✅ Using pre-trained model\n")

    # ---------------------------
    # 1. COMPARISON
    # ---------------------------
    print("📊 EVALUATING PERFORMANCE...\n")

    baseline_cost, rl_cost = compare_with_baseline(model_path)

    print("\n===== FINAL RESULTS =====")
    print(f"Baseline Cost : {baseline_cost:.2f}")
    print(f"RL Agent Cost : {rl_cost:.2f}")

    if baseline_cost != 0:
        savings = ((baseline_cost - rl_cost) / baseline_cost) * 100
        print(f"Cost Reduction: {savings:.2f}%")
    else:
        print("Cost Reduction: ERROR")

    print("==========================\n")

    # ---------------------------
    # 2. PLOTS
    # ---------------------------
    print("📈 GENERATING PLOTS...\n")
    plot_results(baseline_cost, rl_cost, model_path)
    # ---------------------------
    # 3. DEMO
    # ---------------------------
    print("🤖 RUNNING DEMO...\n")
    run_demo(model_path)

    print("\n✅ PIPELINE COMPLETE\n")


if __name__ == "__main__":
    main()