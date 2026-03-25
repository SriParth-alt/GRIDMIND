from src.data_loader import load_datasets, select_buildings
from src.data_preprocessing import preprocess_data
from src.feature_engineering import engineer_features
from src.forecasting_model import train_models
from src.building_agent import BuildingEnergyAgent
from src.grid_manager import GridManagerAgent
from src.sustainability_agent import SustainabilityAgent

import pandas as pd
import matplotlib.pyplot as plt


def main():
    print("Loading datasets...")
    train, building_metadata, weather = load_datasets()

    print("Selecting buildings...")
    train_small, metadata_small = select_buildings(train, building_metadata)

    print("Preprocessing data...")
    processed_data = preprocess_data(train_small, metadata_small, weather)

    print("Engineering features...")
    engineered_data = engineer_features(processed_data)

    print("\nTraining forecasting models...")
    results, best_model = train_models(engineered_data)

    print("\nModel Performance Results:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

    # -----------------------------
    # BAR GRAPH COMPARISON
    # -----------------------------
    model_names = list(results.keys())
    mae_values = [results[m]["MAE"] for m in model_names]
    rmse_values = [results[m]["RMSE"] for m in model_names]
    r2_values = [results[m]["R2"] for m in model_names]

    plt.figure()
    plt.bar(model_names, mae_values)
    plt.title("Model Comparison - MAE")
    plt.ylabel("MAE")
    plt.show()

    plt.figure()
    plt.bar(model_names, rmse_values)
    plt.title("Model Comparison - RMSE")
    plt.ylabel("RMSE")
    plt.show()

    plt.figure()
    plt.bar(model_names, r2_values)
    plt.title("Model Comparison - R2 Score")
    plt.ylabel("R2")
    plt.show()

    print("\n--- ADVANCED MULTI-HOUR MICROGRID SIMULATION ---")

    latest_time = engineered_data['timestamp'].max()
    start_time = latest_time - pd.Timedelta(hours=23)

    simulation_data = engineered_data[
        (engineered_data['timestamp'] >= start_time) &
        (engineered_data['timestamp'] <= latest_time)
    ]

    grid_manager = GridManagerAgent(capacity_limit_ratio=0.9)
    sustainability_agent = SustainabilityAgent()

    before_loads = []
    after_loads = []

    feature_cols = [
        'hour',
        'day_of_week',
        'is_weekend',
        'is_peak_hour',
        'occupancy_ratio',
        'rolling_load_24h',
        'efficiency_ratio',
        'air_temperature',
        'cloud_coverage',
        'dew_temperature',
        'wind_speed'
    ]

    priority_map = {
        0: 1,
        1: 2,
        2: 2,
        3: 3,
        4: 3,
        5: 3
    }

    for timestamp in sorted(simulation_data['timestamp'].unique()):

        hourly_data = simulation_data[
            simulation_data['timestamp'] == timestamp
        ]

        building_agents = []

        for _, row in hourly_data.iterrows():
            building_id = row['building_id']
            priority = priority_map.get(building_id % 6, 2)

            agent = BuildingEnergyAgent(
                building_id=building_id,
                model=best_model,
                priority_level=priority
            )

            feature_row = pd.DataFrame([row[feature_cols]])
            agent.predict_load(feature_row)

            building_agents.append(agent)

        total_before, total_after = grid_manager.apply_grid_control(
            building_agents
        )

        before_loads.append(total_before)
        after_loads.append(total_after)

    metrics = sustainability_agent.summarize(before_loads, after_loads)

    print("\nSimulation Summary (24 Hours):")
    for key, value in metrics.items():
        print(f"{key}: {value:.2f}")

    print("\nBuilding Curtailment Summary:")
    for agent in building_agents:
        print(
            f"Building {agent.building_id} "
            f"Total Curtailed Load: {agent.total_curtailed:.2f}"
        )


if __name__ == "__main__":
    main()