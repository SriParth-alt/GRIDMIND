import pandas as pd
from pathlib import Path

def load_datasets():
    base_path = Path("data/raw")

    train = pd.read_csv(base_path / "train.csv")
    building_metadata = pd.read_csv(base_path / "building_metadata.csv")
    weather = pd.read_csv(base_path / "weather_train.csv")

    return train, building_metadata, weather


def select_buildings(train, building_metadata, num_buildings=6):
    selected_ids = train['building_id'].unique()[:num_buildings]

    filtered_train = train[train['building_id'].isin(selected_ids)]
    filtered_metadata = building_metadata[
        building_metadata['building_id'].isin(selected_ids)
    ]

    return filtered_train, filtered_metadata